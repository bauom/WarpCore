use anthropic::client::{Client as AnthropicClient, ClientBuilder};
use anthropic::types::{ContentBlock, Message, MessagesRequest, Role, MessagesStreamEvent, ContentBlockDelta};
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, CONTENT_TYPE};
use serde::Deserialize;

use inference_lib_core::config::{ApiConfig, BackendConfig, BackendType, GenerationOptions, ModelConfig, ModelType};
use inference_lib_core::error::{InferenceError, Result};
use inference_lib_core::traits::{InferenceService, Model, TextToTextModel};

use async_trait::async_trait;
use futures::StreamExt;
use std::env;
use std::pin::Pin;
use std::sync::Arc;
use tokio_stream::Stream;
use tracing::instrument;

// Environment variables for Anthropic configuration
const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
const ANTHROPIC_API_BASE_ENV: &str = "ANTHROPIC_API_BASE";
const ANTHROPIC_API_VERSION_ENV: &str = "ANTHROPIC_API_VERSION"; // e.g., "2023-06-01"
const ANTHROPIC_DEFAULT_VERSION: &str = "2023-06-01"; // Default version

// Structs for deserializing the /v1/models response
#[derive(Deserialize, Debug)]
struct ModelInfo {
    id: String,
    // Add other fields if needed later (e.g., display_name, type)
}

#[derive(Deserialize, Debug)]
struct ModelsListResponse {
    data: Vec<ModelInfo>,
    // Add other fields if needed (e.g., has_more, first_id, last_id)
}

/// Anthropic Inference Service implementation.
#[derive(Clone)]
pub struct AnthropicService {
    client: Arc<AnthropicClient>,
    #[allow(dead_code)]
    config: ApiConfig,
}

impl AnthropicService {
    #[instrument(skip(config))]
    pub fn new(config: Option<BackendConfig>) -> Result<Self> {
        let api_config = match config {
            Some(BackendConfig::Api(api_conf)) => api_conf,
            Some(BackendConfig::Local(_)) => {
                return Err(InferenceError::InvalidConfig(
                    "Expected API config for Anthropic backend, found Local config".to_string(),
                ));
            }
            None => ApiConfig::new(None),
        };

        let api_key = api_config
            .api_key
            .clone()
            .or_else(|| env::var(ANTHROPIC_API_KEY_ENV).ok())
            .ok_or(InferenceError::ApiKeyMissing(BackendType::Anthropic))?;

        let base_url = api_config
            .base_url
            .clone()
            .or_else(|| env::var(ANTHROPIC_API_BASE_ENV).ok());

        let _api_version = env::var(ANTHROPIC_API_VERSION_ENV).ok()
            .unwrap_or_else(|| "2023-06-01".to_string()); // Default to a known version

        // Build the client using the builder pattern - correct borrowing
        let mut builder = ClientBuilder::default();
        builder.api_key(api_key.clone());
        
        if let Some(base) = base_url.clone() {
            builder.api_base(base);
        }
        
        let client = builder.build()
            .map_err(|e| InferenceError::InvalidConfig(format!("Failed to build Anthropic client: {}", e)))?;

        let service = Self { client: Arc::new(client), config: api_config };
        Ok(service)
    }

    // Helper to get headers for manual requests
    fn get_request_headers(&self) -> Result<HeaderMap> {
        let api_key = self.config.api_key
            .clone()
            .or_else(|| env::var(ANTHROPIC_API_KEY_ENV).ok())
            .ok_or(InferenceError::ApiKeyMissing(BackendType::Anthropic))?;

        let api_version = env::var(ANTHROPIC_API_VERSION_ENV)
            .ok()
            .unwrap_or_else(|| ANTHROPIC_DEFAULT_VERSION.to_string());

        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_str(&api_key)
            .map_err(|e| InferenceError::InvalidConfig(format!("Invalid API key format: {}", e)))?);
        headers.insert("anthropic-version", HeaderValue::from_str(&api_version)
             .map_err(|e| InferenceError::InvalidConfig(format!("Invalid API version format: {}", e)))?);
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(headers)
    }
}

#[async_trait]
impl InferenceService for AnthropicService {
    fn backend_type(&self) -> BackendType {
        BackendType::Anthropic
    }

    fn supported_model_types(&self) -> &[ModelType] {
        &[ModelType::TextToText]
    }

    #[instrument(skip(self, _config))]
    async fn load_model(
        &self,
        model_id_or_path: &str,
        model_type: ModelType,
        _config: Option<ModelConfig>,
    ) -> Result<Arc<dyn Model>> {
        if model_type != ModelType::TextToText {
            return Err(InferenceError::UnsupportedFeature(
                self.backend_type(),
                format!("Model type {:?} not supported", model_type),
            ));
        }

        let model = AnthropicModel {
            client: Arc::clone(&self.client),
            model_id: model_id_or_path.to_string(),
        };

        Ok(Arc::new(model))
    }

    // Anthropic client doesn't have a model listing API
    // async fn list_available_models(&self) -> Result<Vec<String>> { ... }
    #[instrument(skip(self))]
    async fn list_available_models(&self) -> Result<Vec<String>> {
        // Use the base URL from config or default if not present
        let base_url = self.config.base_url
            .clone()
            .or_else(|| env::var(ANTHROPIC_API_BASE_ENV).ok())
            .unwrap_or_else(|| "https://api.anthropic.com".to_string()); // Default Anthropic API base

        let models_url = format!("{}/v1/models", base_url.trim_end_matches('/'));

        let headers = self.get_request_headers()?;

        // Create a reqwest client specifically for this request
        // Ideally, reuse the client from AnthropicClient if possible, but it's not exposed easily in v0.0.8
        let http_client = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .map_err(|e| InferenceError::BackendError(anyhow::anyhow!("Failed to build HTTP client: {}", e)))?;

        let response = http_client.get(&models_url)
            .send()
            .await
            .map_err(|e| InferenceError::ApiRequest(format!("Failed to send request to {}: {}", models_url, e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await.unwrap_or_else(|_| "Failed to read error body".to_string());
             return Err(InferenceError::ApiRequest(format!(
                "API request to {} failed with status {}: {}",
                models_url, status, error_body
            )));
        }

        let models_response = response.json::<ModelsListResponse>()
            .await
            .map_err(|e| InferenceError::ApiRequest(format!("Failed to deserialize models response: {}", e)))?;

        let model_ids = models_response.data.into_iter().map(|m| m.id).collect();

        Ok(model_ids)
    }
}

/// Represents a specific loaded Anthropic model.
pub struct AnthropicModel {
    client: Arc<AnthropicClient>,
    model_id: String,
}

impl Model for AnthropicModel {
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
impl TextToTextModel for AnthropicModel {
    #[instrument(skip(self, prompt, options))]
    async fn generate(
        &self,
        prompt: &str,
        options: Option<GenerationOptions>,
    ) -> Result<String> {
        let opts = options.unwrap_or_default();
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text { text: prompt.to_string() }],
        }];

        let request_body = MessagesRequest {
            model: self.model_id.clone(),
            messages,
            max_tokens: opts.max_tokens.unwrap_or(1024).try_into().unwrap_or(1024),
            stream: false,
            temperature: opts.temperature.map(|f| f as f64),
            top_p: opts.top_p.map(|f| f as f64),
            top_k: opts.top_k,
            stop_sequences: opts.stop_sequences.unwrap_or_default(),
            system: opts.system.unwrap_or_default(),
        };

        let response = self.client.messages(request_body).await?;

        response
            .content
            .iter()
            .find_map(|block| match block {
                ContentBlock::Text { text } => Some(text.clone()),
                _ => None,
            })
            .ok_or_else(|| InferenceError::GenerationError("No text content received from Anthropic".to_string()))
    }

    #[instrument(skip(self, prompt, options))]
    fn generate_stream<'a>(
        &'a self,
        prompt: &'a str,
        options: Option<GenerationOptions>,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        let opts = options.unwrap_or_default();
        let model_id = self.model_id.clone();
        let client = Arc::clone(&self.client);
        let prompt = prompt.to_string();

        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text { text: prompt.to_string() }],
        }];

        let request_body = MessagesRequest {
            model: model_id,
            messages,
            max_tokens: opts.max_tokens.unwrap_or(1024).try_into().unwrap_or(1024),
            stream: true,
            temperature: opts.temperature.map(|f| f as f64),
            top_p: opts.top_p.map(|f| f as f64),
            top_k: opts.top_k,
            stop_sequences: opts.stop_sequences.unwrap_or_default(),
            system: opts.system.unwrap_or_default(),
        };

        let stream = async_stream::try_stream! {
            let mut anthropic_stream = client.messages_stream(request_body).await?;

            while let Some(event_result) = anthropic_stream.next().await {
                match event_result {
                    Ok(event) => {
                        match event {
                            MessagesStreamEvent::ContentBlockDelta { delta, .. } => {
                                match delta {
                                    ContentBlockDelta::TextDelta { text } => yield text,
                                    _ => {} // Ignore other delta types
                                }
                            }
                            _ => {}
                        }
                    }
                    Err(e) => { Err(e)?; }
                }
            }
        };

        Box::pin(stream)
    }
} 