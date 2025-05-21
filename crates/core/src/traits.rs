use crate::error::Result;
use crate::config::{BackendType, ModelType, ModelConfig, GenerationOptions, DiffusionOptions, ImageOutputFormat};
use async_trait::async_trait;
use std::sync::Arc;
use tokio_stream::Stream;
use std::pin::Pin;
use std::path::PathBuf;

/// Represents an inference backend capable of loading models and performing inference.
#[async_trait]
pub trait InferenceService: Send + Sync {
    /// Returns the specific type of this backend.
    fn backend_type(&self) -> BackendType;

    /// Lists the types of models supported by this backend.
    fn supported_model_types(&self) -> &[ModelType];

    /// Loads a model specified by an identifier (path for local, ID for API).
    /// 
    /// Returns a dynamic `Model` trait object.
    async fn load_model(
        &self,
        model_id_or_path: &str,
        model_type: ModelType,
        config: Option<ModelConfig>,
    ) -> Result<Arc<dyn Model>>;

    /// Optionally lists models available from the backend (useful for APIs).
    async fn list_available_models(&self) -> Result<Vec<String>> {
        // Default implementation returns empty or an error
        Ok(vec![])
    }
}

/// A loaded model instance, ready for inference.
/// This is the base trait; specific model types extend this.
pub trait Model: Send + Sync {
    /// Returns the type of this model.
    fn model_type(&self) -> ModelType;

    /// Returns the identifier (path or ID) used to load this model.
    fn name(&self) -> &str;

    /// Provides convenient downcasting to specific model types.
    fn as_text_to_text(&self) -> Option<&dyn TextToTextModel> {
        None
    }
    fn as_text_to_image(&self) -> Option<&dyn TextToImageModel> {
        None
    }
    // fn as_image_to_text(&self) -> Option<&dyn ImageToTextModel> { None }
}

/// A text-to-text generation model (e.g., LLM).
#[async_trait]
pub trait TextToTextModel: Model {
    /// Generates a single text completion for the given prompt.
    async fn generate(
        &self,
        prompt: &str,
        options: Option<GenerationOptions>,
    ) -> Result<String>;

    /// Generates text completion as a stream of tokens.
    fn generate_stream<'a>(
        &'a self,
        prompt: &'a str,
        options: Option<GenerationOptions>,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>>;
}

/// Represents the output of an image generation task.
#[derive(Debug, Clone)]
pub enum ImageOutput {
    File(PathBuf),      // Path to the saved image file
    Bytes(Vec<u8>, ImageOutputFormat), // Raw image bytes and their format
}

/// A text-to-image generation model (e.g., Stable Diffusion).
#[async_trait]
pub trait TextToImageModel: Model {
    /// Generates an image based on the given prompt and options.
    async fn generate_image(
        &self,
        prompt: &str,
        options: Option<DiffusionOptions>,
    ) -> Result<ImageOutput>;
}

// --- Placeholder traits for future model types ---

// #[async_trait]
// pub trait ImageToTextModel: Model {
//     async fn generate(&self, image: /* Image type */, options: Option</* VisionOptions */>) -> Result<String>;
// }

// #[async_trait]
// pub trait TextToImageModel: Model {
//     async fn generate(&self, prompt: &str, options: Option</* DiffusionOptions */>) -> Result</* Image type */>;
// } 