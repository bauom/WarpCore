#[cfg(feature = "diffusion-rs")]
mod diffusion_rs_tests {
    use anyhow::Result;
    use warpcore::{
        create_inference_service, BackendType, DiffusionOptions, ImageOutput, ModelType, SamplerKind,
    };
    use std::env; // For env::var for model path, and temp_dir

    // Helper function to setup tracing and dotenv only once
    fn setup() {
        // Ignore errors if already initialized, or use once_cell for true one-time init
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init();
        dotenv::dotenv().ok();
    }

    #[tokio::test]
    // #[ignore = "Requires diffusion-rs setup and models (preset download or local path)"]
    async fn test_diffusion_preset_generation() -> Result<()> {
        setup();
        println!("--- Testing DiffusionRs Preset Generation ---");

        let backend_type = BackendType::DiffusionRs;
        let service = create_inference_service(backend_type, None).await?;

        let model_id = "SDXLTurbo1_0Fp16"; // Ensure this preset is valid for diffusion-rs v0.1.9
        println!("Loading model from preset: {}", model_id);

        let model = service
            .load_model(model_id, ModelType::TextToImage, None)
            .await?;

        let text_to_image_model = model.as_text_to_image().ok_or_else(|| {
            anyhow::anyhow!("Failed to get TextToImageModel for preset {}", model_id)
        })?;

        let prompt = "A corgi in a wizard hat, detailed, fantasy art";
        let negative_prompt = "blurry, low quality, watermark";
        let temp_dir = env::temp_dir();
        let output_file = temp_dir.join(format!("test_diffusion_preset_{}.png", model_id));

        let options = DiffusionOptions::new()
            .with_width(512) // Smaller for faster test
            .with_height(512)
            .with_steps(8)   // Fewer steps for Turbo
            .with_cfg_scale(1.2)
            .with_negative_prompt(negative_prompt.to_string())
            .with_sampler(SamplerKind::EulerA) // Make sure EulerA is mapped and valid
            .with_seed(12345)
            .with_output_path(output_file.clone());

        println!("Generating image with prompt: '{}'", prompt);
        println!("Output will be saved to: {}", output_file.display());

        match text_to_image_model.generate_image(prompt, Some(options)).await {
            Ok(ImageOutput::File(path)) => {
                println!("Image generation successful! Output at: {}", path.display());
                assert!(path.exists(), "Output file should exist");
            }
            Ok(ImageOutput::Bytes(bytes, format)) => {
                println!("Image generation successful! Received {} bytes in format {:?}.", bytes.len(), format);
                assert!(!bytes.is_empty(), "Byte output should not be empty");
                // Optionally save for inspection during test development
                // std::fs::write(&output_file, bytes)?;
                // println!("Bytes saved to: {}", output_file.display());
            }
            Err(e) => {
                eprintln!("Error during image generation: {:?}", e);
                return Err(e.into()); // Fail the test
            }
        }
        println!("--- Preset generation test finished. ---");
        Ok(())
    }

    #[tokio::test]
    #[ignore = "Requires diffusion-rs setup and a valid local model path."]
    async fn test_diffusion_local_model_generation() -> Result<()> {
        setup();
        println!("--- Testing DiffusionRs Local Model Generation ---");

        // !!! IMPORTANT: Replace with the actual, absolute path to your model file !!!
        let model_file_path_str = "YOUR_HARDCODED_MODEL_PATH_HERE.safetensors";    

        if model_file_path_str == "YOUR_HARDCODED_MODEL_PATH_HERE.safetensors" {
            println!(
                "Skipping local model test: Please update 'model_file_path_str' in diffusion_rs_integration.rs with an actual hardcoded model path."
            );
            return Ok(());
        }
        
        println!("Attempting to load local model from hardcoded path: {}", model_file_path_str);
        // Note: If this is not an absolute path, it will be resolved relative to where the test is run from,
        // or the inference-lib backend might try to resolve it against MODELS_PATH/DIFFUSION_MODELS_PATH.
        // For testing, an absolute path is usually most reliable if not using the env var method.

        let backend_type = BackendType::DiffusionRs;
        let service = create_inference_service(backend_type, None).await?;

        let model = service
            .load_model(&model_file_path_str, ModelType::TextToImage, None)
            .await?;

        let text_to_image_model = model.as_text_to_image().ok_or_else(|| {
            anyhow::anyhow!("Failed to get TextToImageModel for local model {}", model_file_path_str)
        })?;

        let prompt = "A serene forest path at dawn, shafts of light through trees";
        let temp_dir = env::temp_dir();
        let output_file = temp_dir.join("test_diffusion_local_model_output.png");

        let options = DiffusionOptions::new()
            .with_width(512)
            .with_height(512)
            .with_steps(20)
            .with_sampler(SamplerKind::EulerA) // Make sure EulerA is mapped
            .with_seed(54321)
            .with_output_path(output_file.clone());

        println!("Generating image with local model, prompt: '{}'", prompt);
        println!("Output will be saved to: {}", output_file.display());

        match text_to_image_model.generate_image(prompt, Some(options)).await {
            Ok(ImageOutput::File(path)) => {
                println!("Local model image generation successful! Output at: {}", path.display());
                assert!(path.exists(), "Output file should exist");
            }
            Ok(ImageOutput::Bytes(bytes, format)) => {
                println!("Local model image generation successful! Received {} bytes in format {:?}.", bytes.len(), format);
                assert!(!bytes.is_empty(), "Byte output should not be empty");
            }
            Err(e) => {
                eprintln!("Error during local model image generation: {:?}", e);
                return Err(e.into()); // Fail the test
            }
        }
        println!("--- Local model generation test finished. ---");
        Ok(())
    }
} 