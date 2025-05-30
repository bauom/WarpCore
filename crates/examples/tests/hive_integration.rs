// #[cfg(feature = "hive")]
mod hive_tests {
    use anyhow::Result;
    use futures::StreamExt;
    use warpcore::{
        config::{BackendType, GenerationOptions, ModelType},
        create_inference_service,
    };

    fn setup() {
        let _ = tracing_subscriber::fmt::try_init();
        dotenv::dotenv().ok();
    }

    #[tokio::test]
    #[ignore = ""]
    async fn test_hive_list_models() -> Result<()> {
        setup();
        println!("--- Creating Hive Service ---");
        let service = create_inference_service(BackendType::Hive, None, None).await?;

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
    // #[ignore = "Requires NECTAR_TOKEN and running Hive API instance"]
    async fn test_hive_generation_non_streaming() -> Result<()> {
        setup();
        let service = create_inference_service(BackendType::Hive, None, None).await?;

        let model_id = "web:phi-1_5-q4f16_1-MLC";
        let model = service
            .load_model(model_id, ModelType::TextToText, None)
            .await?;

        let text_model = model.as_text_to_text().expect("Model should be TextToText");

        let prompt = "Tell me a joke about programming languages.";
        let options = GenerationOptions::new()
            .with_max_tokens(50)
            .with_temperature(0.6);

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
    #[ignore = "Requires NECTAR_TOKEN and running Hive API instance"]
    async fn test_hive_generation_streaming() -> Result<()> {
        setup();
        let service = create_inference_service(BackendType::Hive, None, None).await?;

        let model_id = "web:phi-1_5-q4f16_1-MLC"; // Adapt to a known working model ID
        let model = service
            .load_model(model_id, ModelType::TextToText, None)
            .await?;

        let text_model = model.as_text_to_text().expect("Model should be TextToText");

        let prompt = "Tell me about saturn, briefly.";
        let options = GenerationOptions::new()
            .with_max_tokens(60)
            .with_temperature(0.7);

        println!("\n--- Performing Streaming Generation ---");
        println!("Prompt: {}", prompt);

        let mut stream = text_model.generate_stream(prompt, Some(options));
        let mut full_response = String::new();
        let mut had_error = false;

        while let Some(chunk) = stream.next().await {
            match chunk {
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
            panic!("Streaming generation failed");
        } else {
            assert!(
                !full_response.is_empty(),
                "Streamed response should not be empty"
            );
        }

        Ok(())
    }
}
