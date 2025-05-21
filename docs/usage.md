# Usage Examples

This document shows common ways to use the `inference-lib` library.

## Prerequisites

1.  **Add Dependency:** Add `inference-lib` to your `Cargo.toml` and enable the features for the backends you want to use:

    ```toml
    [dependencies]
    inference-lib = { version = "0.1.0", features = ["openai"] } # Example: Enable OpenAI
    anyhow = "1.0" # For error handling in examples
    tokio = { version = "1", features = ["full"] } # For async runtime
    futures = "0.3" # For stream handling
    ```

2.  **Configure Backend:** Ensure the necessary configuration (e.g., API keys via environment variables) is set up for your chosen backend. See `backends.md` for details.

## Core Workflow

The typical workflow involves:

1.  **Create Service:** Get an instance of the `InferenceService` trait for your desired backend.
2.  **Load Model:** Load a specific model using the service.
3.  **Get Model Type:** Downcast the generic `Model` trait object to the specific type you need (e.g., `TextToTextModel`).
4.  **Generate:** Call the generation methods (`generate` or `generate_stream`) on the typed model object.

## Example: OpenAI Text Generation

This example demonstrates using the OpenAI backend for both simple and streaming text generation.

```rust
use inference_lib::{
    create_inference_service, BackendType, GenerationOptions, ModelType,
    InferenceService, // Bring the trait into scope
    TextToTextModel, // Bring the trait into scope
};
use futures::StreamExt;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Assumes OPENAI_API_KEY is set in the environment

    // 1. Create Service
    println!("Creating OpenAI service...");
    // Pass None for config to load from environment variables
    let service = create_inference_service(BackendType::OpenAI, None).await?;

    // (Optional) List available models
    // match service.list_available_models().await {
    //     Ok(models) => println!("Available models: {:?}", models),
    //     Err(e) => eprintln!("Could not list models: {}", e),
    // }

    // 2. Load Model
    let model_id = "gpt-3.5-turbo";
    println!("Loading model: {}", model_id);
    let model = service
        .load_model(model_id, ModelType::TextToText, None)
        .await?;

    // 3. Get Model Type
    let text_model = model
        .as_text_to_text()
        .ok_or_else(|| anyhow::anyhow!("Model {} does not support TextToText", model_id))?;

    // --- Non-Streaming Generation ---
    let prompt1 = "Explain the concept of Rust's ownership in one sentence.";
    let options1 = GenerationOptions::new().with_max_tokens(60);

    println!("\n--- Non-Streaming Generation ---");
    println!("Prompt: {}", prompt1);

    // 4. Generate (Non-Streaming)
    match text_model.generate(prompt1, Some(options1)).await {
        Ok(response) => {
            println!("Response: {}", response);
        }
        Err(e) => {
            eprintln!("Generation failed: {}", e);
        }
    }

    // --- Streaming Generation ---
    let prompt2 = "Write a short Rust function that adds two numbers.";
    let options2 = GenerationOptions::new()
        .with_max_tokens(100)
        .with_temperature(0.5);

    println!("\n--- Streaming Generation ---");
    println!("Prompt: {}", prompt2);
    print!("Streamed Response: ");

    // 4. Generate (Streaming)
    let mut stream = text_model.generate_stream(prompt2, Some(options2));
    let mut full_response = String::new();

    while let Some(token_result) = stream.next().await {
        match token_result {
            Ok(token) => {
                print!("{}", token);
                // Flush stdout to ensure tokens appear immediately
                use std::io::{self, Write};
                io::stdout().flush()?;
                full_response.push_str(&token);
            }
            Err(e) => {
                eprintln!("\nStream error: {}", e);
                break; // Stop processing stream on error
            }
        }
    }
    println!(); // Newline after stream finishes

    println!("\nFull streamed response accumulated: {}", full_response);

    Ok(())
}
```

## Using Helper Functions

The library also provides top-level helper functions in `src/lib.rs` for simpler use cases:

-   **`create_inference_service(backend_type, config)`**: As used in the example above, this is the main entry point to get a backend service instance.
-   **`generate_text(backend_type, model_id_or_path, prompt, max_tokens)`**: A quick helper for simple text generation. It handles service creation, model loading, generation, and unloading internally.

```rust
use inference_lib::{generate_text, BackendType};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Assumes OPENAI_API_KEY is set
    let backend = BackendType::OpenAI;
    let model_id = "gpt-3.5-turbo";
    let prompt = "What is the capital of France?";
    let max_tokens = 30;

    println!("Using generate_text helper...");
    match generate_text(backend, model_id, prompt, max_tokens).await {
        Ok(response) => println!("Helper Response: {}", response),
        Err(e) => eprintln!("Helper function failed: {}", e),
    }

    Ok(())
}
```

Refer to the integration tests in `crates/examples/tests/` for more detailed examples specific to each backend (`openai_integration.rs`, `anthropic_integration.rs`, `llama_cpp_integration.rs`). Remember to enable the corresponding features and set up necessary environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `MODELS_PATH`) to run them. 