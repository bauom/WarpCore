use thiserror::Error;

// Add this import if using OpenAIError directly
#[cfg(feature = "openai_error_conversion")]
use async_openai::error::OpenAIError;
#[cfg(feature = "anthropic_error_conversion")]
use anthropic::error::AnthropicError;

pub type Result<T> = std::result::Result<T, InferenceError>;

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum InferenceError {
    #[error("Backend {0:?} is not available or enabled")]
    BackendUnavailable(crate::BackendType),

    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    #[error("Configuration error: {0}")]
    InvalidConfig(String),

    #[error("Generation failed: {0}")]
    GenerationError(String),

    #[error("API request failed: {0}")]
    ApiRequest(String),

    #[error("OpenAI API Error: {0}")]
    OpenAIError(String), // Specific variant for OpenAI errors

    #[error("Anthropic API Error: {0}")]
    AnthropicError(String), // Specific variant for Anthropic errors

    #[error("API key missing or invalid for {0:?}")]
    ApiKeyMissing(crate::BackendType),

    #[error("Feature not supported by backend {0:?}: {1}")]
    UnsupportedFeature(crate::BackendType, String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Internal backend error: {0}")]
    BackendError(#[from] anyhow::Error),

    #[error("An unexpected internal error occurred: {0}")]
    InternalError(String),
}

#[cfg(feature = "openai_error_conversion")]
impl From<OpenAIError> for InferenceError {
    fn from(err: OpenAIError) -> Self {
        // Convert OpenAIError to our specific variant or a general ApiRequest
        // Using a specific variant allows for more granular error handling later.
        InferenceError::OpenAIError(err.to_string())
    }
}

#[cfg(feature = "anthropic_error_conversion")]
impl From<AnthropicError> for InferenceError {
    fn from(err: AnthropicError) -> Self {
        InferenceError::AnthropicError(err.to_string())
    }
}

// Add From impl for the build errors inside the stream macro
// These come from derive_builder usually
impl From<derive_builder::UninitializedFieldError> for InferenceError {
    fn from(err: derive_builder::UninitializedFieldError) -> Self {
        InferenceError::InvalidConfig(format!("Failed to build request options: {}", err))
    }
} 