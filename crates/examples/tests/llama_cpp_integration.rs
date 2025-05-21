#[cfg(feature = "llama_cpp")]
mod llama_cpp_tests {
    use anyhow::Result;
    use futures::StreamExt;
    use warpcore::{
        create_inference_service, BackendType, GenerationOptions, LocalBackendConfig, ModelType, LlamaCppSpecificConfig,
    };
    use std::env;
    use std::io::Write; // For flushing stdout
    use std::path::Path;
    use tracing::info;

    // Define the default model ID used for testing
    const TEST_MODEL_ID: &str = "hf:bartowski/Qwen2-0.5B-Instruct-GGUF:Qwen2-0.5B-Instruct-Q8_0.gguf";

    // Helper function to resolve model path from environment
    fn resolve_model_path(model_id: &str) -> Option<String> {
        let base_path_var = "MODELS_PATH";
        match env::var(base_path_var) {
            Ok(base_path) => {
                // Replace colons with path separators
                let relative_path = model_id.replace(':', "/");
                let full_path = Path::new(&base_path).join(relative_path);
                if full_path.exists() {
                    Some(full_path.to_string_lossy().to_string())
                } else {
                    println!(
                        "Model path {} does not exist under base path {}. Skipping test.",
                        full_path.display(),
                        base_path
                    );
                    None
                }
            }
            Err(_) => {
                println!(
                    "{} environment variable not set. Cannot resolve model path. Skipping test.",
                    base_path_var
                );
                None
            }
        }
    }

    // Helper to initialize tracing (optional)
    fn setup_tracing() {
        dotenv::dotenv().ok(); // Load .env file if present
        // Use try_init to avoid panic if already initialized
        let _ = tracing_subscriber::fmt::try_init(); 
    }

    #[tokio::test]
    #[ignore] // Ignored by default, requires --features llama_cpp and env var
    async fn test_llama_cpp_service_creation() -> Result<()> {
        setup_tracing();
        info!("Testing Llama.cpp service creation...");

        // Fix: Pass LlamaCppSpecificConfig directly, and String directly to with_numa_strategy
        let llama_config = LlamaCppSpecificConfig::default()
            .with_numa_strategy("DISABLED".to_string()); // Pass String directly
        let backend_config = LocalBackendConfig::default()
            .with_llama_cpp_config(llama_config); // Pass LlamaCppSpecificConfig directly

        let service = create_inference_service(
            BackendType::LlamaCpp,
            Some(warpcore::BackendConfig::Local(backend_config)),
        )
        .await?;

        assert_eq!(service.backend_type(), BackendType::LlamaCpp);
        assert!(service.supported_model_types().contains(&ModelType::TextToText));

        info!("Llama.cpp service created successfully.");
        Ok(())
    }

    #[tokio::test]
    #[ignore] // Ignored by default, requires --features llama_cpp and env var
    async fn test_llama_cpp_model_loading() -> Result<()> {
        setup_tracing();
        // Check if model exists first, but don't pass the path to load_model
        let model_path_resolved = match resolve_model_path(TEST_MODEL_ID) {
            Some(p) => p,
            None => return Ok(()), // Skip test
        };
        info!(model_id=%TEST_MODEL_ID, path=%model_path_resolved, "Checking Llama.cpp model loading with ID...");

        let service = create_inference_service(BackendType::LlamaCpp, None).await?;
        // Load using the Model ID
        let model = service
            .load_model(TEST_MODEL_ID, ModelType::TextToText, None)
            .await?;

        // Check if the model name matches the ID used
        assert_eq!(model.name(), TEST_MODEL_ID);
        assert_eq!(model.model_type(), ModelType::TextToText);

        info!("Llama.cpp model loaded successfully using ID.");
        Ok(())
    }

    #[tokio::test]
    #[ignore] // Ignored by default, requires --features llama_cpp and env var
    async fn test_llama_cpp_generate() -> Result<()> {
        setup_tracing();
        // Check if model exists first
        let model_path_resolved = match resolve_model_path(TEST_MODEL_ID) {
            Some(p) => p,
            None => return Ok(()), // Skip test
        };
        info!(model_id=%TEST_MODEL_ID, path=%model_path_resolved, "Checking Llama.cpp generate with ID...");

        let service = create_inference_service(BackendType::LlamaCpp, None).await?;
        // Load using the Model ID
        let model = service
            .load_model(TEST_MODEL_ID, ModelType::TextToText, None)
            .await?;
        let text_model = model.as_text_to_text().expect("Model should be TextToText");

        let prompt = "Explain the concept of 'generative AI' in one sentence.";
        // Fix: Pass u32 directly to with_max_tokens
        let options = GenerationOptions::default().with_max_tokens(50); // Pass 50 directly

        let response = text_model.generate(prompt, Some(options)).await?;

        info!(prompt = prompt, response = response, "Generation complete.");
        assert!(!response.is_empty(), "Generated response should not be empty");

        Ok(())
    }

    #[tokio::test]
    #[ignore] // Ignored by default, requires --features llama_cpp and env var
    async fn test_llama_cpp_generate_stream() -> Result<()> {
        setup_tracing();
        // Check if model exists first
        let model_path_resolved = match resolve_model_path(TEST_MODEL_ID) {
            Some(p) => p,
            None => return Ok(()), // Skip test
        };
        info!(model_id=%TEST_MODEL_ID, path=%model_path_resolved, "Checking Llama.cpp generate_stream with ID...");

        let service = create_inference_service(BackendType::LlamaCpp, None).await?;
         // Load using the Model ID
        let model = service
            .load_model(TEST_MODEL_ID, ModelType::TextToText, None)
            .await?;
        let text_model = model.as_text_to_text().expect("Model should be TextToText");

        let prompt = "Write a short Rust function that adds two numbers.";
        // Fix: Pass u32 directly to with_max_tokens
        let options = GenerationOptions::default().with_max_tokens(100); // Pass 100 directly

        let mut stream = text_model.generate_stream(prompt, Some(options));
        let mut response_text = String::new();
        let mut token_count = 0;

        print!("Stream: ");
        while let Some(token_res) = stream.next().await {
            match token_res {
                Ok(token) => {
                    print!("{}", token);
                    std::io::stdout().flush()?;
                    response_text.push_str(&token);
                    token_count += 1;
                }
                Err(e) => {
                    // Print error and fail test
                    eprintln!("\nError during stream: {:?}", e);
                    return Err(e.into());
                }
            }
        }
        println!(); // Newline after stream

        info!(prompt = prompt, response = response_text, tokens = token_count, "Streaming complete.");
        assert!(token_count > 0, "Should receive at least one token");
        assert!(!response_text.is_empty(), "Streamed response should not be empty");

        Ok(())
    }

    #[tokio::test]
    #[ignore] // Ignored by default, requires --features llama_cpp and env var
    async fn test_llama_cpp_list_models() -> Result<()> {
        setup_tracing();
        info!("Testing Llama.cpp list_available_models...");
        // This test requires MODELS_PATH to be set and contain the TEST_MODEL_ID path
        let expected_model_id = TEST_MODEL_ID;
        match resolve_model_path(expected_model_id) {
            Some(_) => info!(id = %expected_model_id, "Test model path resolved successfully. Proceeding with list test."),
            None => {
                info!("Test model path could not be resolved. Skipping list models test.");
                return Ok(()) // Skip test if model path isn't set up correctly
            }
        }
        let service = create_inference_service(BackendType::LlamaCpp, None).await?;
        let models = service.list_available_models().await?;

        info!(models = ?models, "Listed models.");

        assert!(!models.is_empty(), "Model list should not be empty when MODELS_PATH is set and contains models.");
        
        // Check if the original TEST_MODEL_ID is present in the results
        let expected_model_id = TEST_MODEL_ID;
        assert!(
            models.contains(&expected_model_id.to_string()),
            "Expected model ID '{}' not found in listed models: {:?}",
            expected_model_id,
            models
        );

        info!("Llama.cpp list models test passed.");
        Ok(())
    }
} 