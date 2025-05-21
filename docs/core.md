# Core Abstractions

The `warpcore-core` crate defines the fundamental building blocks for the library. It provides the traits, configuration structs, and error types used by all backend implementations.

## Key Components

### 1. Traits (`traits.rs`)

These define the capabilities expected from backends and models.

-   **`InferenceService`**: Represents a connection to an inference backend (e.g., OpenAI API, a local Llama.cpp instance).
    -   `backend_type() -> BackendType`: Returns the type of the backend.
    -   `supported_model_types() -> &[ModelType]`: Lists the model types (e.g., `TextToText`) supported.
    -   `load_model(...) -> Result<Arc<dyn Model>>`: Loads a model by its identifier (path or ID) and type. Returns a `Model` trait object.
    -   `list_available_models() -> Result<Vec<String>>`: (Optional) Lists models available on the backend.

-   **`Model`**: Represents a loaded model instance.
    -   `model_type() -> ModelType`: Returns the type of the model.
    -   `name() -> &str`: Returns the identifier used to load the model.
    -   `as_text_to_text() -> Option<&dyn TextToTextModel>`: Helper for downcasting to a specific model type trait. (Similar helpers planned for other types).

-   **`TextToTextModel`**: Extends `Model` for text generation capabilities.
    -   `generate(...) -> Result<String>`: Generates a single completion.
    -   `generate_stream(...) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>>`: Generates a stream of token completions.

### 2. Configuration (`config.rs`)

These structs define how to configure backends, models, and generation.

-   **`BackendType`** (enum): Identifies the backend (e.g., `OpenAI`, `Anthropic`, `LlamaCpp`).
-   **`ModelType`** (enum): Identifies the model type (e.g., `TextToText`).
-   **`BackendConfig`** (enum): Holds backend-specific configuration.
    -   `Api(ApiConfig)`: For API-based backends.
    -   `Local(LocalBackendConfig)`: For local backends.
-   **`ApiConfig`**: Configuration for API backends.
    -   `api_key: Option<String>`
    -   `organization: Option<String>`
    -   `base_url: Option<String>`
    -   `timeout: Option<Duration>`
-   **`LocalBackendConfig`**: Configuration for local backends.
    -   `threads: Option<u32>`
    -   `use_gpu: Option<bool>`
    -   `gpu_layers: Option<u32>`
    -   `llama_cpp: Option<LlamaCppSpecificConfig>` (Nested specific config)
-   **`LlamaCppSpecificConfig`**: Llama.cpp specific settings (mmap, mlock, numa).
-   **`ModelConfig`**: Configuration applied when loading a specific model.
    -   `context_size: Option<u32>`
-   **`GenerationOptions`**: Parameters controlling the generation process.
    -   `max_tokens: Option<u32>`
    -   `temperature: Option<f32>`
    -   `top_p: Option<f32>`
    -   `top_k: Option<usize>`
    -   `stop_sequences: Option<Vec<String>>`
    -   `system: Option<String>` (System prompt)

Most configuration structs provide builder-style methods (e.g., `.with_max_tokens(100)`).

### 3. Error Handling (`error.rs`)

-   **`InferenceError`** (enum): A unified error type covering various issues like configuration errors, model loading failures, generation errors, API errors, missing keys, etc.
-   **`Result<T>`**: An alias for `std::result::Result<T, InferenceError>`.

`InferenceError` includes `From` implementations for common errors (like `std::io::Error`, `anyhow::Error`) and, via feature flags (`openai_error_conversion`, `anthropic_error_conversion`), can automatically convert errors from specific backend libraries (e.g., `async_openai::error::OpenAIError`) into `InferenceError` variants (e.g., `InferenceError::OpenAIError`). 