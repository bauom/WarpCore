use async_openai::config::OpenAIConfig;
use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs,
};
use async_openai::Client as OpenAIClient;
use warpcore_core::config::GenerationOptions;
use warpcore_core::error::{InferenceError, Result};
use warpcore_core::traits::{InferenceService, Model, TextToTextModel};
use warpcore_core::{ApiConfig, BackendConfig, BackendType, ModelConfig, ModelType};

use async_trait::async_trait;
use futures::StreamExt;
use std::env;
use std::pin::Pin;
use std::sync::Arc;
use tokio_stream::Stream;
use tracing::instrument;

// Environment variables for OpenAI configuration
const OPENAI_API_KEY_ENV: &str = "OPENAI_API_KEY";
const OPENAI_ORG_ID_ENV: &str = "OPENAI_ORGANIZATION";
const OPENAI_API_BASE_ENV: &str = "OPENAI_API_BASE";

/// OpenAI Inference Service implementation.
#[derive(Clone)]
pub struct OpenAIService {
    client: OpenAIClient<OpenAIConfig>,
    #[allow(dead_code)] // Config might be used later
    config: ApiConfig, // Store original config for reference if needed
}

impl OpenAIService {
    /// Creates a new OpenAIService.
    ///
    /// Reads configuration from `BackendConfig::Api` or environment variables.
    #[instrument(skip(config))]
    pub fn new(config: Option<BackendConfig>) -> Result<Self> {
        let api_config = match config {
            Some(BackendConfig::Api(api_conf)) => api_conf,
            Some(BackendConfig::Local(_)) => {
                return Err(InferenceError::InvalidConfig(
                    "Expected API config for OpenAI backend, found Local config".to_string(),
                ));
            }
            None => ApiConfig::new(None), // Create default, will load from env
        };

        // Load API key: Config > Env
        let api_key = api_config
            .api_key
            .clone()
            .or_else(|| env::var(OPENAI_API_KEY_ENV).ok())
            .ok_or(InferenceError::ApiKeyMissing(BackendType::OpenAI))?;

        // Load Org ID: Config > Env
        let org_id = api_config
            .organization
            .clone()
            .or_else(|| env::var(OPENAI_ORG_ID_ENV).ok());

        // Load Base URL: Config > Env > Default
        let base_url = api_config
            .base_url
            .clone()
            .or_else(|| env::var(OPENAI_API_BASE_ENV).ok());

        let mut openai_config = OpenAIConfig::new().with_api_key(api_key);
        if let Some(org) = org_id {
            openai_config = openai_config.with_org_id(org);
        }
        if let Some(base) = base_url {
            openai_config = openai_config.with_api_base(base);
        }

        // Create custom reqwest client for timeout
        let mut http_client_builder = reqwest::ClientBuilder::new();
        if let Some(timeout) = api_config.timeout {
            http_client_builder = http_client_builder.timeout(timeout);
        }
        // Consider adding other reqwest settings like user-agent here
        let http_client = http_client_builder.build().map_err(|e| {
            InferenceError::InvalidConfig(format!("Failed to build HTTP client: {}", e))
        })?;

        // Pass custom client to OpenAIClient
        let client = OpenAIClient::with_config(openai_config).with_http_client(http_client);

        Ok(Self {
            client,
            config: api_config, // Store the resolved config
        })
    }
}

#[async_trait]
impl InferenceService for OpenAIService {
    fn backend_type(&self) -> BackendType {
        BackendType::OpenAI
    }

    fn supported_model_types(&self) -> &[ModelType] {
        &[ModelType::TextToText]
    }

    #[instrument(skip(self, _config))]
    async fn load_model(
        &self,
        model_id_or_path: &str,
        model_type: ModelType,
        _config: Option<ModelConfig>, // OpenAI doesn't use ModelConfig currently
    ) -> Result<Arc<dyn Model>> {
        if model_type != ModelType::TextToText {
            return Err(InferenceError::UnsupportedFeature(
                self.backend_type(),
                format!("Model type {:?} not supported", model_type),
            ));
        }

        // For OpenAI, loading is just creating the model handle
        let model = OpenAIModel {
            client: self.client.clone(),
            model_id: model_id_or_path.to_string(),
        };

        Ok(Arc::new(model))
    }

    #[instrument(skip(self))]
    async fn list_available_models(&self) -> Result<Vec<String>> {
        Ok(self
            .client
            .models()
            .list()
            .await
            .map_err(|e| InferenceError::ApiRequest(e.to_string()))?
            .data
            .into_iter()
            .map(|m| m.id)
            .collect())
    }
}

/// Represents a specific loaded OpenAI model.
pub struct OpenAIModel {
    client: OpenAIClient<OpenAIConfig>,
    model_id: String,
}

impl Model for OpenAIModel {
    fn model_type(&self) -> ModelType {
        ModelType::TextToText
    }

    fn name(&self) -> &str {
        &self.model_id
    }

    fn as_text_to_text(&self) -> Option<&dyn TextToTextModel> {
        Some(self)
    }
}

#[async_trait]
impl TextToTextModel for OpenAIModel {
    #[instrument(skip(self, prompt, options))]
    async fn generate(&self, prompt: &str, options: Option<GenerationOptions>) -> Result<String> {
        let opts = options.unwrap_or_default();

        // Basic prompt templating (user role)
        let messages = vec![ChatCompletionRequestUserMessageArgs::default()
            .content(prompt)
            .build()
            .map_err(|e| InferenceError::InvalidConfig(format!("Failed to build prompt: {}", e)))?
            .into()];

        let mut request_builder = CreateChatCompletionRequestArgs::default();
        request_builder.model(&self.model_id).messages(messages);

        // Map GenerationOptions to OpenAI request parameters
        if let Some(max_tokens_u32) = opts.max_tokens {
            let max_tokens_u16: u16 = max_tokens_u32.try_into().map_err(|_| {
                InferenceError::InvalidConfig(format!(
                    "max_tokens value {} is too large for OpenAI API (max 65535)",
                    max_tokens_u32
                ))
            })?;
            request_builder.max_tokens(max_tokens_u16);
        }
        if let Some(temp) = opts.temperature {
            request_builder.temperature(temp);
        }
        if let Some(top_p) = opts.top_p {
            request_builder.top_p(top_p);
        }
        // Note: top_k is not directly supported in OpenAI Chat API, use top_p
        if let Some(stop) = opts.stop_sequences {
            request_builder.stop(stop);
        }
        // Add other mappings as needed (presence_penalty, frequency_penalty, etc.)

        let request = request_builder.build().map_err(|e| {
            InferenceError::InvalidConfig(format!("Failed to build request: {}", e))
        })?;

        let response = self.client.chat().create(request).await?;

        // Extract the first choice's message content
        response
            .choices
            .get(0)
            .and_then(|choice| choice.message.content.clone())
            .ok_or_else(|| {
                InferenceError::GenerationError("No content received from OpenAI".to_string())
            })
    }

    #[instrument(skip(self, prompt, options))]
    fn generate_stream<'a>(
        &'a self,
        prompt: &'a str,
        options: Option<GenerationOptions>,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        let opts = options.unwrap_or_default();
        let model_id = self.model_id.clone();
        let client = self.client.clone();
        let prompt = prompt.to_string(); // Clone prompt for async block

        // Build the user message first, handling potential errors
        let user_message_result = ChatCompletionRequestUserMessageArgs::default()
            .content(prompt)
            .build() // This returns Result<ChatCompletionRequestUserMessage, OpenAIError>
            .map_err(|e| InferenceError::OpenAIError(e.to_string())); // Map error type

        // Decide whether to proceed or return an error stream
        let messages = match user_message_result {
            Ok(user_message) => {
                // Conversion is infallible after successful build
                let request_message: ChatCompletionRequestMessage = user_message.into();
                vec![request_message] // Proceed with this vector
            }
            Err(e) => {
                // If message building failed, return a stream that yields just the error
                return Box::pin(tokio_stream::once(Err(e)));
            }
        };

        // Now, create the stream, knowing messages are valid.
        let stream = async_stream::try_stream! {
            let mut request_builder = CreateChatCompletionRequestArgs::default();
            request_builder
                .model(model_id)
                .messages(messages) // Use the successfully built messages
                .stream(true);

            // Map GenerationOptions...
            if let Some(max_tokens_u32) = opts.max_tokens {
                let max_tokens_u16: u16 = max_tokens_u32.try_into().map_err(|_| {
                     InferenceError::InvalidConfig(format!(
                        "max_tokens value {} is too large for OpenAI API (max 65535)",
                        max_tokens_u32
                    ))
                })?;
                request_builder.max_tokens(max_tokens_u16);
            }
            if let Some(temp) = opts.temperature {
                request_builder.temperature(temp);
            }
             if let Some(top_p) = opts.top_p {
                request_builder.top_p(top_p);
            }
             if let Some(stop) = opts.stop_sequences {
                request_builder.stop(stop);
            }

            let request = request_builder.build()?; // ? is okay here (derive_builder error)

            let mut openai_stream = client.chat().create_stream(request).await?; // ? is okay here (OpenAIError)

            while let Some(chunk_result) = openai_stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        if let Some(delta_content) = chunk.choices.get(0).and_then(|c| c.delta.content.as_ref()) {
                            yield delta_content.clone();
                        }
                    }
                    Err(e) => {
                        Err(e)?;
                    }
                }
            }
        };

        Box::pin(stream)
    }
}
