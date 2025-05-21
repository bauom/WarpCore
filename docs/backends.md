# Backend Integrations

This library supports multiple inference backends through separate crates located in `crates/backends/`. Each backend implements the `InferenceService` trait from `warpcore-core`.

To use a specific backend, you need to enable its corresponding feature flag when depending on `warpcore`.

```toml
# Example in your Cargo.toml
[dependencies]
warpcore = { version = "0.1.0", features = ["openai", "anthropic"] } # Enable OpenAI and Anthropic
```

The main `warpcore` crate re-exports the necessary types and functions when a feature is enabled.

## Available Backends

### 1. OpenAI (`features = ["openai"]`)

-   **Crate:** `crates/backends/openai` (`warpcore-openai`)
-   **Service:** `OpenAIService`
-   **Type:** `BackendType::OpenAI`
-   **Config:** `BackendConfig::Api(ApiConfig)`
-   **Core Dependency:** `async-openai`
-   **Description:** Integrates with the OpenAI API (ChatGPT, GPT-4, etc.).
-   **Configuration:**
    -   Uses `ApiConfig`.
    -   Reads API key from `ApiConfig.api_key` or `OPENAI_API_KEY` environment variable (required).
    -   Reads organization ID from `ApiConfig.organization` or `OPENAI_ORGANIZATION` environment variable (optional).
    -   Reads base URL from `ApiConfig.base_url` or `OPENAI_API_BASE` environment variable (optional, defaults to OpenAI API).
    -   Supports timeout via `ApiConfig.timeout`.
-   **Models:** Can list available models using `list_available_models()`.

### 2. Anthropic (`features = ["anthropic"]`)

-   **Crate:** `crates/backends/anthropic` (`warpcore-anthropic`)
-   **Service:** `AnthropicService`
-   **Type:** `BackendType::Anthropic`
-   **Config:** `BackendConfig::Api(ApiConfig)`
-   **Core Dependency:** `anthropic` (Rust SDK)
-   **Description:** Integrates with the Anthropic API (Claude models).
-   **Configuration:**
    -   Uses `ApiConfig`.
    -   Reads API key from `ApiConfig.api_key` or `ANTHROPIC_API_KEY` environment variable (required).
    -   Reads base URL from `ApiConfig.base_url` or `ANTHROPIC_API_BASE` environment variable (optional, defaults to Anthropic API).
    -   Reads API version from `ANTHROPIC_API_VERSION` environment variable (optional, defaults to `2023-06-01`).
-   **Models:** Can list available models using `list_available_models()` (uses a direct HTTP request).

### 3. Llama.cpp (`features = ["llama_cpp"]`)

-   **Crate:** `crates/backends/llama_cpp` (`warpcore-llama_cpp`)
-   **Service:** `LlamaCppService`
-   **Type:** `BackendType::LlamaCpp`
-   **Config:** `BackendConfig::Local(LocalBackendConfig)`
-   **Core Dependency:** `llama-cpp-rs`
-   **Description:** Integrates with local models via the `llama.cpp` library. Requires `llama.cpp` to be built and available.
-   **Configuration:**
    -   Uses `LocalBackendConfig`.
    -   Reads general local settings like `threads`, `use_gpu`, `gpu_layers`.
    -   Reads Llama.cpp specific settings from `LocalBackendConfig.llama_cpp` (`LlamaCppSpecificConfig`) like `use_mmap`, `use_mlock`, `numa_strategy`.
    -   **Requires `MODELS_PATH` environment variable:** This variable must point to a directory containing the GGUF model files. The `model_id_or_path` passed to `load_model` is interpreted relative to this path (e.g., `hf:bartowski/Qwen2-0.5B-Instruct-GGUF:Qwen2-0.5B-Instruct-Q8_0.gguf` resolves to `$MODELS_PATH/hf/bartowski/Qwen2-0.5B-Instruct-GGUF/Qwen2-0.5B-Instruct-Q8_0.gguf`).
-   **Models:** Does not support `list_available_models()`. Model paths must be known and provided.

## Backend Configuration (`BackendConfig`)

When creating a service (e.g., via `create_inference_service`), you provide an `Option<BackendConfig>`:

-   **`None`**: The backend will attempt to load its configuration entirely from environment variables (e.g., API keys).
-   **`Some(BackendConfig::Api(api_config))`**: Provide an `ApiConfig` struct. Fields within `ApiConfig` (like `api_key`) that are `None` will still try to fall back to environment variables.
-   **`Some(BackendConfig::Local(local_config))`**: Provide a `LocalBackendConfig` struct for local backends.

See `core.md` for details on the `ApiConfig` and `LocalBackendConfig` structs. 