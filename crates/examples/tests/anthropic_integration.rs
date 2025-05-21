#[cfg(feature = "anthropic")]
mod anthropic_tests {
    use anyhow::Result;
    use futures::StreamExt;
    use inference_lib::{create_inference_service, BackendType, GenerationOptions, ModelType};
    use std::env;

    // Helper function to setup tracing and dotenv only once
    fn setup() {
        // Ignore errors if already initialized
        let _ = tracing_subscriber::fmt::try_init();
        dotenvy::dotenv().ok();
    }

    #[tokio::test]
    #[ignore = "Requires ANTHROPIC_API_KEY and network access"]
    async fn test_anthropic_list_models() -> anyhow::Result<()> {
        setup();
        println!("--- Creating Anthropic Service ---");
        let service = create_inference_service(BackendType::Anthropic, None).await?;

        println!("--- Listing Available Models ---");
        match service.list_available_models().await {
            Ok(models) => {
                println!("Available models: {:?}", models);
                assert!(!models.is_empty(), "Should return at least one model");
            }
            Err(e) => {
                eprintln!("Failed to list models: {}", e);
                panic!("Model listing failed: {}", e);
            }
        }
        Ok(())
    }

    #[tokio::test]
    #[ignore = "Requires ANTHROPIC_API_KEY and network access"]
    async fn test_anthropic_generation_non_streaming() -> anyhow::Result<()> {
        setup();
        let service = create_inference_service(BackendType::Anthropic, None).await?;
        
        // *** IMPORTANT: Replace with a valid model ID for your account ***
        let model_id = "claude-3-haiku-20240307"; 

        println!("\n--- Loading Model: {} ---", model_id);
        let model = service
            .load_model(model_id, ModelType::TextToText, None)
            .await?;
        let text_model = model
            .as_text_to_text()
            .expect("Model should be TextToText");

        let prompt = "Write a short, funny story about a conversation between a Rust crab and a Python snake.";
        let options = GenerationOptions::new()
            .with_max_tokens(300) // Anthropic requires max_tokens
            .with_temperature(0.8);

        println!("\n--- Performing Non-Streaming Generation ---");
        println!("Prompt: {}", prompt);
        match text_model.generate(prompt, Some(options.clone())).await {
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
    #[ignore = "Requires ANTHROPIC_API_KEY and network access"]
    async fn test_anthropic_generation_streaming() -> anyhow::Result<()> {
        setup();
        let service = create_inference_service(BackendType::Anthropic, None).await?;

        // *** IMPORTANT: Replace with a valid model ID for your account ***
        let model_id = "claude-3-haiku-20240307"; 

        let model = service
            .load_model(model_id, ModelType::TextToText, None)
            .await?;
        let text_model = model
            .as_text_to_text()
            .expect("Model should be TextToText");

        let prompt = "Write a short, funny story about a conversation between a Rust crab and a Python snake.";
        let options = GenerationOptions::new()
            .with_max_tokens(300)
            .with_temperature(0.8);

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