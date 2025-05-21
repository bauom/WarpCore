#[cfg(feature = "openai")]
mod openai_tests {
    // Remove unused imports
    // use warpcore::*;
    // use tokio_stream::StreamExt;
    use anyhow::Result;
    use futures::StreamExt; // Keep this one if actually used by stream processing below
    use warpcore::{create_inference_service, BackendType, GenerationOptions, ModelType};
    use std::env;

    // Helper function to setup tracing and dotenv only once
    fn setup() {
        // Ignore errors if already initialized
        let _ = tracing_subscriber::fmt::try_init();
        dotenvy::dotenv().ok();
    }

    #[tokio::test]
    #[ignore = "Requires OPENAI_API_KEY and network access"]
    async fn test_openai_list_models() -> anyhow::Result<()> {
        setup();
        println!("--- Creating OpenAI Service ---");
        let service = create_inference_service(BackendType::OpenAI, None).await?;

        println!("--- Listing Available Models ---");
        match service.list_available_models().await {
            Ok(models) => {
                println!("Available models: {:?}", models);
                assert!(!models.is_empty(), "Should return at least one model");
            }
            Err(e) => {
                eprintln!("Failed to list models: {}", e);
                panic!("Model listing failed: {}", e); // Fail the test
            }
        }
        Ok(())
    }

    #[tokio::test]
    #[ignore = "Requires OPENAI_API_KEY and network access"]
    async fn test_openai_generation_non_streaming() -> anyhow::Result<()> {
        setup();
        let service = create_inference_service(BackendType::OpenAI, None).await?;
        let model_id = "gpt-3.5-turbo"; // Or another suitable model
        println!("\n--- Loading Model: {} ---", model_id);
        let model = service
            .load_model(model_id, ModelType::TextToText, None)
            .await?;
        let text_model = model
            .as_text_to_text()
            .expect("Model should be TextToText");

        let prompt = "Write a short haiku about Rust programming.";
        let options = GenerationOptions::new()
            .with_max_tokens(50)
            .with_temperature(0.7);

        println!("\n--- Performing Non-Streaming Generation ---");
        println!("Prompt: {}", prompt);
        match text_model.generate(prompt, Some(options)).await {
            Ok(response) => {
                println!("\nResponse:\n{}", response);
                assert!(!response.is_empty(), "Response should not be empty");
            }
            Err(e) => {
                eprintln!("Generation failed: {}", e);
                panic!("Non-streaming generation failed: {}", e);
            }
        }
        Ok(())
    }

    #[tokio::test]
    #[ignore = "Requires OPENAI_API_KEY and network access"]
    async fn test_openai_generation_streaming() -> anyhow::Result<()> {
        setup();
        let service = create_inference_service(BackendType::OpenAI, None).await?;
        let model_id = "gpt-3.5-turbo"; // Or another suitable model
        let model = service
            .load_model(model_id, ModelType::TextToText, None)
            .await?;
        let text_model = model
            .as_text_to_text()
            .expect("Model should be TextToText");

        let prompt = "Write a short haiku about asynchronous Rust.";
        let options = GenerationOptions::new()
            .with_max_tokens(50)
            .with_temperature(0.7);

        println!("\n--- Performing Streaming Generation ---");
        println!("Prompt: {}", prompt);
        print!("\nStreamed Response: ");
        let mut stream = text_model.generate_stream(prompt, Some(options));
        let mut full_response = String::new();
        let mut had_error = false;

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
                    had_error = true;
                    break;
                }
            }
        }
        println!(); // Newline after stream finishes

        if had_error {
            panic!("Streaming generation failed.");
        } else {
            assert!(!full_response.is_empty(), "Streamed response should not be empty");
        }

        Ok(())
    }
} 