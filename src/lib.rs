// This file will re-export items based on enabled features

pub use warpcore_core::*;

#[cfg(feature = "openai")]
pub use warpcore_openai;

#[cfg(feature = "llama_cpp")]
pub use warpcore_llama_cpp;

#[cfg(feature = "anthropic")]
pub use warpcore_anthropic;

#[cfg(feature = "diffusion-rs")]
pub use warpcore_diffusion_rs;

// --- Top-level helper functions ---

use std::sync::Arc;

/// Convenience function to create an inference service based on type and config.
/// Requires the corresponding feature flag to be enabled.
pub async fn create_inference_service(
    backend_type: BackendType,
    config: Option<BackendConfig>,
) -> Result<Arc<dyn InferenceService>> {
    match backend_type {
        #[cfg(feature = "openai")]
        BackendType::OpenAI => {
            let service = warpcore_openai::OpenAIService::new(config)?;
            Ok(Arc::new(service))
        }
        #[cfg(feature = "llama_cpp")]
        BackendType::LlamaCpp => {
            use warpcore_llama_cpp::LlamaCppService;
            let service = LlamaCppService::new(config)?;
            Ok(Arc::new(service))
        }
        #[cfg(feature = "anthropic")]
        BackendType::Anthropic => {
            let service = warpcore_anthropic::AnthropicService::new(config)?;
            Ok(Arc::new(service))
        }
        #[cfg(feature = "diffusion-rs")]
        BackendType::DiffusionRs => {
            let service = warpcore_diffusion_rs::DiffusionRsService::new(config)?;
            Ok(Arc::new(service))
        }
        _ => Err(InferenceError::BackendUnavailable(backend_type)),
    }
}

/// Quick helper for text generation using a specified backend.
/// Loads the model, generates text, and unloads implicitly.
pub async fn generate_text(
    backend_type: BackendType,
    model_id_or_path: &str,
    prompt: &str,
    max_tokens: u32, // Simplified options for this helper
    // Consider adding Option<BackendConfig> here too
) -> Result<String> {
    let service = create_inference_service(backend_type, None).await?;
    let model = service
        .load_model(model_id_or_path, ModelType::TextToText, None)
        .await?;

    let text_model = model
        .as_text_to_text()
        .ok_or_else(|| InferenceError::UnsupportedFeature(
            backend_type,
            "TextToText model type".to_string()
        ))?;

    let options = GenerationOptions::new().with_max_tokens(max_tokens);

    text_model.generate(prompt, Some(options)).await
} 