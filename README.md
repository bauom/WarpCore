# WarpCore – Wildcard AI Runtime & Pipeline Connector Orchestrated in a Rust Engine

A modular Rust workspace for running inference with **local LLM engines** (e.g. `llama.cpp`), **local image generation models** (e.g., via `diffusion-rs`), and **cloud AI APIs** (e.g. OpenAI, Anthropic).

> **Status:** Actively developed, core API stabilizing. Backends for OpenAI, Anthropic, Llama.cpp and Diffusion-rs are functional.

---

## Why?

Rust developers shouldn't rewrite glue-code every time they swap models or providers. This crate offers a **unified, type-safe interface** for loading models, streaming tokens, generating images, and switching backends with *one* line of code. Use OpenAI today, switch to a local model or Anthropic tomorrow, or generate images with Stable Diffusion.

```rust
let service = create_inference_service(
    BackendType::Anthropic, // or BackendType::OpenAI, BackendType::LlamaCpp (planned), ...
    None // Optional backend-specific config (e.g., API keys)
).await?;

let model = service.load_model(
    "claude-3-haiku-20240307", // Cloud model ID or local model path
    ModelType::TextToText,
    None // Optional model-specific config
).await?;

let text_model = model.as_text_to_text().unwrap();

// Simple generation
let response = text_model.generate("Explain Rust's borrow checker simply.", None).await?;
println!("Response: {}", response);

// Streaming generation
let mut stream = text_model.generate_stream("Write a Rust program that counts to 5.", None);
print!("Stream: ");
while let Some(token_res) = stream.next().await {
    print!("{}", token_res?);
    std::io::stdout().flush()?; // Ensure tokens appear immediately
}
println!();

// Text-to-image example
let image_service = create_inference_service(
    BackendType::DiffusionRs,
    None
).await?;
let image_model_desc = image_service.list_available_models().await?.first().cloned().unwrap_or_else(|| "SDXLTurbo1_0Fp16".to_string());
let image_model = image_service.load_model(
    &image_model_desc,
    ModelType::TextToImage,
    None
).await?;
let text_to_image_model = image_model.as_text_to_image().unwrap();
let image_options = DiffusionOptions::new().with_steps(10).with_width(512).with_height(512);
match text_to_image_model.generate_image("photo of a rusty robot, cinematic lighting", Some(image_options)).await? {
    ImageOutput::File(path) => println!("Image saved to: {}", path.display()),
    ImageOutput::Bytes(_, format) => println!("Image generated in memory as {:?}", format),
}
```

---

## Features

- Trait-based abstraction over backends (`InferenceService`) and model types (`TextToTextModel`, `TextToImageModel`, etc.)
- Async/await & streaming generation (`Stream<Item = Result<String>>`) for text.
- Image generation with configurable options (`DiffusionOptions`, `ImageOutput`).
- Ergonomic config objects (`BackendConfig`, `ApiConfig`, `GenerationOptions`, `DiffusionOptions`, ...)
- Unified error handling (`InferenceError` with `From` conversions)
- First-class support for **local** and **cloud** engines
- Optional dependencies via Cargo feature flags (`--features openai,anthropic,llama_cpp,diffusion-rs`)
- Integration tests for core functionality (runnable with API keys)

---

## Workspace Layout

```
crates/
├─ core/          # Core traits, configs, error types
├─ backends/
│   ├─ llama_cpp/   # (Planned)
│   ├─ openai/
│   ├─ anthropic/
│   └─ diffusion-rs/ # Image generation
└─ examples/      # Integration tests (formerly examples)
    └─ tests/
```

Each backend lives in its own crate so you only compile what you need.

---

## Supported Backends

| Backend                    | Status      | Features Needed | Dependency         | Notes                                    |
|----------------------------|-------------|-----------------|--------------------|------------------------------------------|
| `openai` (cloud)           | ✅ Available | `openai`        | `async-openai`     | Text-to-text                             |
| `anthropic` (cloud)        | ✅ Available | `anthropic`     | `anthropic-rs`   | Text-to-text                             |
| `llama-cpp` (local)        | ✅ Available | `llama_cpp`     | `llama-cpp-2`    | Text-to-text, GGUF models                |
| `diffusion-rs` (local)     | ✅ Available | `diffusion-rs`  | `diffusion-rs`     | Text-to-image, Stable Diffusion models   |
| `candle-rs` (local)        | 💤 Planned  | `candle`        |                    |                                          |
| `groq` / `modal` (cloud)   | 💤 Planned  |                 |                    |                                          |

---

## Supported Model Types

| Model Type       | Status      |
|------------------|-------------|
| `TextToText`     | ✅ Available |
| `TextToImage`    | ✅ Available |
| `ImageToText`    | 💤 Planned  |

Currently, only `TextToText` and `TextToImage` models are supported.

---

## Integration Tests

Integration tests for the implemented backends are available in the `crates/examples` directory. They require API keys and network access and are ignored by default.

You can run them using:

```bash
# Ensure API keys are set in environment or .env file
# e.g., export OPENAI_API_KEY=sk-...
#       export ANTHROPIC_API_KEY=sk-...

# Run OpenAI tests
cargo test --package inference-lib-examples --features openai -- --ignored

# Run Anthropic tests
cargo test --package inference-lib-examples --features anthropic -- --ignored

# Run Llama.cpp tests
# Requires the MODELS_PATH environment variable pointing to a directory containing models.
# The test expects the model 'hf:bartowski/Qwen2-0.5B-Instruct-GGUF:Qwen2-0.5B-Instruct-Q8_0.gguf'
# to exist at '$MODELS_PATH/hf/bartowski/Qwen2-0.5B-Instruct-GGUF/Qwen2-0.5B-Instruct-Q8_0.gguf'.
# export MODELS_PATH=/path/to/your/models/directory
# NOTE: Due to potential concurrency issues in the underlying library, run with --test-threads=1
cargo test --package inference-lib-examples --features llama_cpp -- --ignored --test-threads=1

# Run Diffusion-rs tests
# The preset test (SDXLTurbo1_0Fp16) runs by default and downloads model weights.
# The local model test is ignored by default; requires DIFFUSION_MODELS_PATH (or MODELS_PATH)
# and a model file (e.g., *.safetensors, *.gguf) at that location.
# export DIFFUSION_MODELS_PATH=/path/to/your/diffusion/models
cargo test --package inference-lib-examples --features diffusion-rs --test diffusion_rs_integration

# Run all available ignored tests (requires all relevant API keys and model paths)
# Note: Llama.cpp tests require MODELS_PATH and the specific test model.
# It's recommended to run 'all' tests sequentially if including llama_cpp:
# cargo test --package inference-lib-examples --features all -- --ignored --test-threads=1
cargo test --package inference-lib-examples --features all -- --ignored
```

---

## Architecture (high-level)

1.  **`InferenceService`** trait – Implemented by each backend (`OpenAIService`, `AnthropicService`, `DiffusionRsService`). Responsible for listing and loading models.
2.  **`Model`** trait – Represents a loaded model instance. Can be downcast using helper methods (e.g., `as_text_to_text`, `as_text_to_image`).
3.  **Model Type Traits** (`TextToTextModel`, `TextToImageModel`, etc.) – Define specific generation capabilities (e.g., `generate`, `generate_stream`, `generate_image`).
4.  **Configuration Structs** (`BackendConfig`, `ApiConfig`, `GenerationOptions`, `DiffusionOptions`, etc.) – Provide typed configuration for backends and generation.
5.  **`InferenceError`** enum – Centralized error type with conversions for backend-specific errors.
6.  **Top-level helpers** (`create_inference_service`) provide convenience wrappers.

---

## Configuration & Secrets

Backend configuration can be provided explicitly or loaded automatically from environment variables.

```rust
use inference_lib::{create_inference_service, BackendType, BackendConfig, ApiConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Explicit config
    let api_cfg = ApiConfig::new(Some("sk-...".to_string())) // Or load from env if None
                  .with_base_url("https://api.example.com");
    let backend_cfg = BackendConfig::Api(api_cfg);
    let service = create_inference_service(BackendType::OpenAI, Some(backend_cfg)).await?;

    // Auto-load from environment
    let service_auto = create_inference_service(BackendType::Anthropic, None).await?;

    Ok(())
}

```

The library looks for the following environment variables:

| Backend   | Environment Keys                                             |
|-----------|--------------------------------------------------------------|
| OpenAI    | `OPENAI_API_KEY`, `OPENAI_ORGANIZATION`, `OPENAI_API_BASE`   |
| Anthropic | `ANTHROPIC_API_KEY`, `ANTHROPIC_API_BASE`, `ANTHROPIC_API_VERSION` |
| Llama.cpp | `MODELS_PATH` (for resolving relative model paths)           |
| Diffusion-rs | `DIFFUSION_MODELS_PATH`, `MODELS_PATH` (for resolving relative model paths) |
| *Groq*    | *`GROQ_API_KEY`* (Planned)                                   |

---

## Observability

Basic `tracing` spans are emitted for key operations. You can integrate with `tracing-subscriber` or OpenTelemetry for detailed production monitoring. (More detailed instrumentation planned).

---

## License

MIT.

---

For planned milestones and open tasks, check **ROADMAP.md**. 