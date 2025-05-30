use serde_json::json;
use tokio::time::sleep;
use warpcore_core::config::GenerationOptions;
use warpcore_core::error::{InferenceError, Result};
use warpcore_core::traits::{InferenceService, Model, TextToTextModel};
use warpcore_core::{ApiConfig, BackendConfig, BackendType, ModelConfig, ModelType};

use async_stream::stream;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio_stream::{self as stream, Stream};
use tracing::instrument;

const HIVE_URL: &str = "http://localhost:3000";

#[derive(Clone)]
pub struct HiveApiService {
    client: HiveApiClient,
    // config: ApiConfig,
}

#[derive(Clone)]
struct HiveApiClient {
    base_url: String,
    nectar: String,
    cluster_id: Option<String>,
}

struct HiveModel {
    client: HiveApiClient,
    model_id: String,
}

impl Model for HiveModel {
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
impl TextToTextModel for HiveModel {
    async fn generate(&self, prompt: &str, _options: Option<GenerationOptions>) -> Result<String> {
        let dto = CreateInferenceDTO {
            prompt: prompt.to_string(),
            modelID: self.model_id.to_string(),
            params: Some({
                let mut map = HashMap::new();
                map.insert("repeatLastN".to_string(), json!(64));
                map.insert("repeatPenalty".to_string(), json!(1.1));
                map.insert("sampleLen".to_string(), json!(4096));
                map.insert("seed".to_string(), json!(5));
                map.insert("temperature".to_string(), json!(0.8));
                map.insert("topP".to_string(), json!(0.95));
                map
            }),
            clusterId: self.client.cluster_id.clone(),
        };

        let url = format!("{}/v1/inference", self.client.base_url);
        let client = reqwest::Client::new();
        let res = client
            .post(&url)
            .header("x-aios-nectar", &self.client.nectar)
            .json(&dto)
            .send()
            .await
            .map_err(|e| InferenceError::ApiRequest(e.to_string()))?;

        if !res.status().is_success() {
            return Err(InferenceError::ApiRequest(format!(
                "Failed with status: {}",
                res.status()
            )));
        }

        let id = res
            .json::<AddInferenceResponseDTO>()
            .await
            .map_err(|e| InferenceError::ApiRequest(e.to_string()))?
            .id;

        let polling_url = format!("{}/v1/inference/{}", self.client.base_url, id);

        loop {
            let poll_res = client
                .get(&polling_url)
                .header("x-aios-nectar", &self.client.nectar)
                .send()
                .await
                .map_err(|e| InferenceError::ApiRequest(e.to_string()))?;

            if !poll_res.status().is_success() {
                return Err(InferenceError::ApiRequest(format!(
                    "Polling failed with status: {}",
                    poll_res.status()
                )));
            }

            let parsed = poll_res
                .json::<GetInferenceByIdResponseDTO>()
                .await
                .map_err(|e| InferenceError::ApiRequest(e.to_string()))?;

            match parsed.status.as_str() {
                "COMPLETED" => {
                    return parsed.result.ok_or_else(|| {
                        InferenceError::GenerationError("Completed but no result returned".into())
                    });
                }
                "FAILED" | "CANCELLED" => {
                    let reason = parsed
                        .errorMessage
                        .unwrap_or_else(|| "Unknown failure".to_string());
                    return Err(InferenceError::GenerationError(reason));
                }
                "PENDING" => {
                    sleep(Duration::from_secs(1)).await;
                }
                _ => {
                    return Err(InferenceError::ApiRequest(
                        "Unexpected status from polling".into(),
                    ));
                }
            }
        }
    }

    fn generate_stream<'a>(
        &'a self,
        prompt: &'a str,
        _options: Option<GenerationOptions>,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        let prompt = prompt.to_string();
        let model_id = self.model_id.clone();
        let base_url = self.client.base_url.clone();
        let nectar = self.client.nectar.clone();

        Box::pin(stream! {
            let dto = CreateInferenceDTO {
                prompt: prompt.clone(),
                modelID: model_id.clone(),
                params: Some({
                    let mut map = HashMap::new();
                    map.insert("repeatLastN".to_string(), json!(64));
                    map.insert("repeatPenalty".to_string(), json!(1.1));
                    map.insert("sampleLen".to_string(), json!(4096));
                    map.insert("seed".to_string(), json!(5));
                    map.insert("temperature".to_string(), json!(0.8));
                    map.insert("topP".to_string(), json!(0.95));
                    map
                }),
                clusterId: None,
            };

            let client = reqwest::Client::new();
            let url = format!("{}/v1/inference", base_url);

            let res = client
                .post(&url)
                .header("x-aios-nectar", &nectar)
                .json(&dto)
                .send()
                .await;

            let id = match res {
                Ok(resp) => {
                    if !resp.status().is_success() {
                        yield Err(InferenceError::ApiRequest(format!("Failed with status: {}", resp.status())));
                        return;
                    }
                    match resp.json::<AddInferenceResponseDTO>().await {
                        Ok(parsed) => parsed.id,
                        Err(e) => {
                            yield Err(InferenceError::ApiRequest(e.to_string()));
                            return;
                        }
                    }
                },
                Err(e) => {
                    yield Err(InferenceError::ApiRequest(e.to_string()));
                    return;
                }
            };

            let mut previous = String::new();
            let polling_url = format!("{}/v1/inference/{}", base_url, id);
            loop {
                let poll_res = client
                    .get(&polling_url)
                    .header("x-aios-nectar", &nectar)
                    .send()
                    .await;

                match poll_res {
                    Ok(res) => {
                        if !res.status().is_success() {
                            yield Err(InferenceError::ApiRequest(format!("Polling failed with status: {}", res.status())));
                            break;
                        }
                        let parsed = res.json::<GetInferenceByIdResponseDTO>().await;
                        match parsed {
                            Ok(resp) => {
                                if let Some(ref result) = resp.result {
                                    if result.len() > previous.len() {
                                        let new = &result[previous.len()..];
                                        previous = result.clone();
                                        yield Ok(new.to_string());
                                    }
                                }

                                match resp.status.as_str() {
                                    "PENDING" => {
                                        sleep(Duration::from_secs(1)).await;
                                        continue;
                                    }


                                    "COMPLETED" => break,
                                    "FAILED" | "CANCELLED" => {
                                        let reason = resp.errorMessage.unwrap_or("Unknown failure".into());
                                        yield Err(InferenceError::GenerationError(reason));
                                        break;
                                    }
                                    _ => {
                                        yield Err(InferenceError::ApiRequest("Unexpected status".into()));
                                        break;
                                    }
                                }
                            }
                            Err(e) => {
                                yield Err(InferenceError::ApiRequest(e.to_string()));
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(InferenceError::ApiRequest(e.to_string()));
                        break;
                    }
                }
            }
        })
    }
}

impl HiveApiService {
    pub fn new(config: Option<BackendConfig>, cluster_id: Option<String>) -> Result<Self> {
        let api_config = match config {
            Some(BackendConfig::Api(api_conf)) => api_conf,
            Some(BackendConfig::Local(_)) => {
                return Err(InferenceError::InvalidConfig(
                    "Expected API config for Hive backend, found Local config".to_string(),
                ));
            }
            None => ApiConfig::new(None), // Create default, will load from env
        };

        let hive_url = api_config.base_url.unwrap_or_else(|| {
            std::env::var("HIVE_URL").expect("Missing HIVE_URL in config and env")
        });

        let nectar_token = api_config.api_key.unwrap_or_else(|| {
            std::env::var("NECTAR_TOKEN").expect("Missing NECTAR_TOKEN in config and env")
        });

        Ok(Self {
            client: HiveApiClient {
                base_url: hive_url,
                nectar: nectar_token,
                cluster_id: cluster_id.clone(),
            },
            // config: None,
        })
    }
}

#[async_trait]
impl InferenceService for HiveApiService {
    fn backend_type(&self) -> BackendType {
        BackendType::Hive
    }

    fn supported_model_types(&self) -> &[ModelType] {
        &[ModelType::TextToText]
    }

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

        let model = HiveModel {
            client: self.client.clone(),
            model_id: model_id_or_path.to_string(),
        };

        Ok(Arc::new(model))
    }

    async fn list_available_models(&self) -> Result<Vec<String>> {
        let mut url = format!("{}/v2/models/live", self.client.base_url);
        let nectar = &self.client.nectar;
        let mut query = vec![];
        query.push(("take", "100".to_string()));

        if !query.is_empty() {
            url = format!("{}?{}", url, serde_urlencoded::to_string(query).unwrap());
        }

        let client = reqwest::Client::new();
        let res = client
            .get(&url)
            .header("x-aios-nectar", nectar)
            .send()
            .await
            .map_err(|e| InferenceError::ApiRequest(e.to_string()))?;

        if !res.status().is_success() {
            return Err(InferenceError::ApiRequest(format!(
                "Failed with status: {}",
                res.status()
            )));
        }

        let data = res
            .json::<ListLiveModelsResponseDTOV2>()
            .await
            .map_err(|e| InferenceError::ApiRequest(e.to_string()))?;

        Ok(data
            .models
            .into_iter()
            .filter_map(|m| m.model.id.into())
            .collect())
    }
}

// =========================================== structs ==========================================

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateInferenceDTO {
    prompt: String,
    modelID: String,
    #[serde(default)]
    params: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    clusterId: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AddInferenceResponseDTO {
    id: String,
    userPK: String,
    modelID: String,
    prompt: String,
    params: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct ListLiveModelsResponseDTOV2 {
    models: Vec<LiveModelEntityV2>,
    count: Option<u32>,
    total: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct LiveModelEntityV2 {
    model: ModelEntityV2,
    availableNodes: u32,
    activeNodes: u32,
    state: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelEntityV2 {
    id: String,
    name: Option<String>,
    repository: Option<String>,
    ggufFile: Option<String>,
    hf: Option<bool>,
    #[serde(default)]
    hyper: Option<bool>,
    parameters: Option<String>,
    size: Option<u64>,
    tokenizerFile: Option<String>,
    torrent: Option<String>,
    createdAt: String,
    updatedAt: String,
    verified: bool,
    uri: Option<String>,
    hash: Option<String>,
    version: String,
    template: Option<String>,
    allowed: bool,
    fileURI: Option<String>,
    #[serde(rename = "type")]
    model_type: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GetInferenceByIdResponseDTO {
    pub id: String,
    pub userPK: String,
    pub modelID: String,
    pub prompt: String,
    pub params: serde_json::Value, // You can define a proper `InferenceRequestParams` struct if needed
    pub status: String,            // Consider using an enum if you want stricter typing
    pub createdAt: u64,

    #[serde(default)]
    pub nodePK: Option<String>,
    #[serde(default)]
    pub registrationID: Option<String>,
    #[serde(default)]
    pub result: Option<String>,
    #[serde(default)]
    pub completedAt: Option<u64>,
    #[serde(default)]
    pub attempts: Option<u32>,
    #[serde(default)]
    pub tokenStats: Option<serde_json::Value>, // Replace with concrete type if available
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub errorMessage: Option<String>,
}
