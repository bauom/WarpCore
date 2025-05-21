use async_trait::async_trait;
use warpcore_core::{
    config::{
        BackendConfig, BackendType, DiffusionOptions, ImageOutputFormat, ModelConfig as CoreModelConfig, ModelType, SamplerKind,
    },
    error::{InferenceError, Result},
    traits::{ImageOutput, InferenceService, Model, TextToImageModel},
};
// Note: diffusion_rs structs are named similarly, qualify them to avoid ambiguity
use diffusion_rs::{
    api::{
        Config as DiffusionRsApiConfig, // Renamed to avoid clash
        ConfigBuilder as DiffusionRsApiConfigBuilder, // Renamed
        ModelConfig as DiffusionRsModelConfig, // Renamed
        ModelConfigBuilder as DiffusionRsModelConfigBuilder, // Renamed
        SampleMethod as DiffusionRsSampleMethod, // Renamed
        txt2img, // Direct import
        WeightType as DiffusionRsWeightType, // Added for preset mapping
    },
    preset::{Preset as DiffusionRsPreset, PresetBuilder as DiffusionRsPresetBuilder}, // Renamed
};
use std::{
    env,
    path::{Path, PathBuf},
    sync::Arc,
};
use strum_macros::EnumString; // For parsing preset strings
use tracing::{debug, error, info, instrument};

// Environment variable for model paths
const DIFFUSION_MODELS_PATH_ENV: &str = "DIFFUSION_MODELS_PATH";
const DEFAULT_MODELS_PATH_ENV: &str = "MODELS_PATH"; // Fallback

/// Maps our SamplerKind to diffusion_rs::api::SampleMethod
fn map_to_diffusion_rs_sampler(sampler: Option<SamplerKind>) -> Option<DiffusionRsSampleMethod> {
    sampler.map(|s| match s {
        SamplerKind::EulerA => DiffusionRsSampleMethod::EULER_A,
        SamplerKind::Euler => DiffusionRsSampleMethod::EULER,
        SamplerKind::Heun => DiffusionRsSampleMethod::HEUN,
        SamplerKind::Dpm2 => DiffusionRsSampleMethod::DPM2,
        SamplerKind::Dpmpp2sA => DiffusionRsSampleMethod::DPMPP2S_A,
        SamplerKind::Dpmpp2m => DiffusionRsSampleMethod::DPMPP2M,
        SamplerKind::Dpmpp2mv2 => DiffusionRsSampleMethod::DPMPP2Mv2,
        SamplerKind::Ipndm => DiffusionRsSampleMethod::IPNDM,
        SamplerKind::IpndmV => DiffusionRsSampleMethod::IPNDM_V,
        SamplerKind::Lcm => DiffusionRsSampleMethod::LCM,
        SamplerKind::DdimTrailing => DiffusionRsSampleMethod::DDIM_TRAILING,
        SamplerKind::Tcd => DiffusionRsSampleMethod::TCD,
        // SamplerKind variants LMS, Dpm2A, DpmppSde, UniPc, Pndm (original) are not in diffusion_rs v0.1.9 SampleMethod
        // Add a catch-all or handle them explicitly if they should map to a default or error
        // For now, unmapped SamplerKind variants from our core lib will cause a panic here.
        // Consider adding an explicit error or default mapping for these.
        // _ => todo!("Sampler mapping needed: SamplerKind::{:?}. Check diffusion-rs v0.1.9 SampleMethod enum for correct variants.", s),
    })
}

// fn map_from_diffusion_rs_sampler(sampler: DiffusionRsSampleMethod) -> SamplerKind {
//     match sampler {
//         DiffusionRsSampleMethod::EULER_A => SamplerKind::EulerA,
//         DiffusionRsSampleMethod::EULER => SamplerKind::Euler,
//         DiffusionRsSampleMethod::HEUN => SamplerKind::Heun,
//         DiffusionRsSampleMethod::DPM2 => SamplerKind::Dpm2,
//         DiffusionRsSampleMethod::DPMPP2S_A => SamplerKind::Dpmpp2sA,
//         DiffusionRsSampleMethod::DPMPP2M => SamplerKind::Dpmpp2m,
//         DiffusionRsSampleMethod::DPMPP2Mv2 => SamplerKind::Dpmpp2mv2,
//         DiffusionRsSampleMethod::IPNDM => SamplerKind::Ipndm,
//         DiffusionRsSampleMethod::IPNDM_V => SamplerKind::IpndmV,
//         DiffusionRsSampleMethod::LCM => SamplerKind::Lcm,
//         DiffusionRsSampleMethod::DDIM_TRAILING => SamplerKind::DdimTrailing,
//         DiffusionRsSampleMethod::TCD => SamplerKind::Tcd,
//         DiffusionRsSampleMethod::N_SAMPLE_METHODS => {
//             todo!("Reverse mapping for N_SAMPLE_METHODS not well-defined for SamplerKind")
//         }
//         // Since diffusion_rs_sys::sample_method_t is non_exhaustive, we need a wildcard arm.
//         _ => todo!("Unhandled diffusion_rs::SampleMethod variant: {:?}. Update SamplerKind and mapping.", sampler),
//     }
// }

/// A helper to parse preset strings that diffusion-rs might not parse directly
/// This aligns with the `map_str_to_preset_test` from the user's example.
#[derive(Debug, Clone, Copy, EnumString)]
#[strum(ascii_case_insensitive)]
enum InternalPresetMapping {
    SD1_5Fp16,
    SD1_5Fp32,
    SDXLFp16,
    SDXLFp32,
    SDXLTurbo1_0Fp16,
}

impl From<InternalPresetMapping> for DiffusionRsPreset {
    fn from(val: InternalPresetMapping) -> Self {
        match val {
            InternalPresetMapping::SD1_5Fp16 | InternalPresetMapping::SD1_5Fp32 => DiffusionRsPreset::StableDiffusion1_5,
            InternalPresetMapping::SDXLFp16 | InternalPresetMapping::SDXLFp32 => DiffusionRsPreset::SDXLBase1_0,
            InternalPresetMapping::SDXLTurbo1_0Fp16 => DiffusionRsPreset::SDXLTurbo1_0Fp16,
        }
    }
}

// --- Start of new preset mapping logic ---

fn weight_type_to_string(wt: DiffusionRsWeightType) -> String {
    match wt {
        DiffusionRsWeightType::SD_TYPE_F32 => "F32".to_string(),
        DiffusionRsWeightType::SD_TYPE_F16 => "F16".to_string(),
        DiffusionRsWeightType::SD_TYPE_Q4_0 => "Q4_0".to_string(),
        DiffusionRsWeightType::SD_TYPE_Q4_1 => "Q4_1".to_string(),
        DiffusionRsWeightType::SD_TYPE_Q5_0 => "Q5_0".to_string(),
        DiffusionRsWeightType::SD_TYPE_Q5_1 => "Q5_1".to_string(),
        DiffusionRsWeightType::SD_TYPE_Q8_0 => "Q8_0".to_string(),
        DiffusionRsWeightType::SD_TYPE_Q8_1 => "Q8_1".to_string(),
        DiffusionRsWeightType::SD_TYPE_Q2_K => "Q2K".to_string(),
        DiffusionRsWeightType::SD_TYPE_Q3_K => "Q3K".to_string(),
        DiffusionRsWeightType::SD_TYPE_Q4_K => "Q4K".to_string(),
        DiffusionRsWeightType::SD_TYPE_Q5_K => "Q5K".to_string(),
        DiffusionRsWeightType::SD_TYPE_Q6_K => "Q6K".to_string(),
        DiffusionRsWeightType::SD_TYPE_Q8_K => "Q8K".to_string(),
        DiffusionRsWeightType::SD_TYPE_IQ2_XXS => "IQ2XXS".to_string(),
        DiffusionRsWeightType::SD_TYPE_IQ2_XS => "IQ2XS".to_string(),
        DiffusionRsWeightType::SD_TYPE_IQ3_XXS => "IQ3XXS".to_string(),
        DiffusionRsWeightType::SD_TYPE_IQ1_S => "IQ1S".to_string(),
        DiffusionRsWeightType::SD_TYPE_IQ4_NL => "IQ4NL".to_string(),
        DiffusionRsWeightType::SD_TYPE_IQ3_S => "IQ3S".to_string(),
        DiffusionRsWeightType::SD_TYPE_IQ2_S => "IQ2S".to_string(),
        DiffusionRsWeightType::SD_TYPE_IQ4_XS => "IQ4XS".to_string(),
        DiffusionRsWeightType::SD_TYPE_I8 => "I8".to_string(),
        DiffusionRsWeightType::SD_TYPE_I16 => "I16".to_string(),
        DiffusionRsWeightType::SD_TYPE_I32 => "I32".to_string(),
        DiffusionRsWeightType::SD_TYPE_I64 => "I64".to_string(),
        DiffusionRsWeightType::SD_TYPE_F64 => "F64".to_string(),
        DiffusionRsWeightType::SD_TYPE_IQ1_M => "IQ1M".to_string(),
        DiffusionRsWeightType::SD_TYPE_BF16 => "BF16".to_string(),
        // Explicitly handle COUNT or add a catch-all if necessary
        _ => format!("UnsupportedWeightType{:?}", wt),
    }
}

fn string_to_weight_type(s: &str) -> Option<DiffusionRsWeightType> {
    match s.to_uppercase().as_str() {
        "F32" => Some(DiffusionRsWeightType::SD_TYPE_F32),
        "F16" => Some(DiffusionRsWeightType::SD_TYPE_F16),
        "Q4_0" | "Q40" => Some(DiffusionRsWeightType::SD_TYPE_Q4_0),
        "Q4_1" | "Q41" => Some(DiffusionRsWeightType::SD_TYPE_Q4_1),
        "Q5_0" | "Q50" => Some(DiffusionRsWeightType::SD_TYPE_Q5_0),
        "Q5_1" | "Q51" => Some(DiffusionRsWeightType::SD_TYPE_Q5_1),
        "Q8_0" | "Q80" => Some(DiffusionRsWeightType::SD_TYPE_Q8_0),
        "Q8_1" | "Q81" => Some(DiffusionRsWeightType::SD_TYPE_Q8_1),
        "Q2K" => Some(DiffusionRsWeightType::SD_TYPE_Q2_K),
        "Q3K" => Some(DiffusionRsWeightType::SD_TYPE_Q3_K),
        "Q4K" => Some(DiffusionRsWeightType::SD_TYPE_Q4_K),
        "Q5K" => Some(DiffusionRsWeightType::SD_TYPE_Q5_K),
        "Q6K" => Some(DiffusionRsWeightType::SD_TYPE_Q6_K),
        "Q8K" => Some(DiffusionRsWeightType::SD_TYPE_Q8_K),
        "IQ2XXS" => Some(DiffusionRsWeightType::SD_TYPE_IQ2_XXS),
        "IQ2XS" => Some(DiffusionRsWeightType::SD_TYPE_IQ2_XS),
        "IQ3XXS" => Some(DiffusionRsWeightType::SD_TYPE_IQ3_XXS),
        "IQ1S" => Some(DiffusionRsWeightType::SD_TYPE_IQ1_S),
        "IQ4NL" => Some(DiffusionRsWeightType::SD_TYPE_IQ4_NL),
        "IQ3S" => Some(DiffusionRsWeightType::SD_TYPE_IQ3_S),
        "IQ2S" => Some(DiffusionRsWeightType::SD_TYPE_IQ2_S),
        "IQ4XS" => Some(DiffusionRsWeightType::SD_TYPE_IQ4_XS),
        "I8" => Some(DiffusionRsWeightType::SD_TYPE_I8),
        "I16" => Some(DiffusionRsWeightType::SD_TYPE_I16),
        "I32" => Some(DiffusionRsWeightType::SD_TYPE_I32),
        "I64" => Some(DiffusionRsWeightType::SD_TYPE_I64),
        "F64" => Some(DiffusionRsWeightType::SD_TYPE_F64),
        "IQ1M" => Some(DiffusionRsWeightType::SD_TYPE_IQ1_M),
        "BF16" => Some(DiffusionRsWeightType::SD_TYPE_BF16),
        _ => None,
    }
}

fn drs_preset_to_string(preset: DiffusionRsPreset) -> String {
    match preset {
        DiffusionRsPreset::StableDiffusion1_4 => "StableDiffusion1_4".to_string(),
        DiffusionRsPreset::StableDiffusion1_5 => "StableDiffusion1_5".to_string(),
        DiffusionRsPreset::StableDiffusion2_1 => "StableDiffusion2_1".to_string(),
        DiffusionRsPreset::StableDiffusion3MediumFp16 => "StableDiffusion3MediumFp16".to_string(),
        DiffusionRsPreset::SDXLBase1_0 => "SDXLBase1_0".to_string(),
        DiffusionRsPreset::SDTurbo => "SDTurbo".to_string(),
        DiffusionRsPreset::SDXLTurbo1_0Fp16 => "SDXLTurbo1_0Fp16".to_string(),
        DiffusionRsPreset::StableDiffusion3_5MediumFp16 => "StableDiffusion3_5MediumFp16".to_string(),
        DiffusionRsPreset::StableDiffusion3_5LargeFp16 => "StableDiffusion3_5LargeFp16".to_string(),
        DiffusionRsPreset::StableDiffusion3_5LargeTurboFp16 => "StableDiffusion3_5LargeTurboFp16".to_string(),
        DiffusionRsPreset::JuggernautXL11 => "JuggernautXL11".to_string(),
        // Variants with WeightType
        DiffusionRsPreset::Flux1Dev(wt) => format!("Flux1Dev_{}", weight_type_to_string(wt)),
        DiffusionRsPreset::Flux1Schnell(wt) => format!("Flux1Schnell_{}", weight_type_to_string(wt)),
        DiffusionRsPreset::Flux1Mini(wt) => format!("Flux1Mini_{}", weight_type_to_string(wt)),
        // Handle any new variants that might be added to the non_exhaustive enum
        _ => {
            // Attempt to create a somewhat descriptive name, though it might not be perfect
            // without knowing the exact structure of future variants.
            // For now, a generic placeholder or logging an error might be suitable.
            // This debug representation is a fallback.
            format!("UnknownOrUnsupportedPreset_{:?}", preset)
        }
    }
}

fn string_to_drs_preset(s: &str) -> Option<DiffusionRsPreset> {
    match s {
        "StableDiffusion1_4" => Some(DiffusionRsPreset::StableDiffusion1_4),
        "StableDiffusion1_5" => Some(DiffusionRsPreset::StableDiffusion1_5),
        "StableDiffusion2_1" => Some(DiffusionRsPreset::StableDiffusion2_1),
        "StableDiffusion3MediumFp16" => Some(DiffusionRsPreset::StableDiffusion3MediumFp16),
        "SDXLBase1_0" => Some(DiffusionRsPreset::SDXLBase1_0),
        "SDTurbo" => Some(DiffusionRsPreset::SDTurbo),
        "SDXLTurbo1_0Fp16" => Some(DiffusionRsPreset::SDXLTurbo1_0Fp16),
        "StableDiffusion3_5MediumFp16" => Some(DiffusionRsPreset::StableDiffusion3_5MediumFp16),
        "StableDiffusion3_5LargeFp16" => Some(DiffusionRsPreset::StableDiffusion3_5LargeFp16),
        "StableDiffusion3_5LargeTurboFp16" => Some(DiffusionRsPreset::StableDiffusion3_5LargeTurboFp16),
        "JuggernautXL11" => Some(DiffusionRsPreset::JuggernautXL11),
        _ => { // Handle variants with WeightType
            if let Some((prefix, wt_str)) = s.rsplit_once('_') {
                string_to_weight_type(wt_str).and_then(|wt| match prefix {
                    "Flux1Dev" => Some(DiffusionRsPreset::Flux1Dev(wt)),
                    "Flux1Schnell" => Some(DiffusionRsPreset::Flux1Schnell(wt)),
                    "Flux1Mini" => Some(DiffusionRsPreset::Flux1Mini(wt)),
                    _ => None,
                })
            } else {
                None
            }
        }
    }
}
// --- End of new preset mapping logic ---

#[derive(Clone)]
pub struct DiffusionRsService;

impl DiffusionRsService {
    pub fn new(_config: Option<BackendConfig>) -> Result<Self> {
        // BackendConfig might be used later for GPU selection, thread counts, etc.
        // For now, diffusion-rs handles its own internal setup.
        info!("DiffusionRsService initialized");
        Ok(Self)
    }
}

#[async_trait]
impl InferenceService for DiffusionRsService {
    fn backend_type(&self) -> BackendType {
        BackendType::DiffusionRs
    }

    fn supported_model_types(&self) -> &[ModelType] {
        &[ModelType::TextToImage]
    }

    #[instrument(skip(self), fields(model_id_or_path = %model_id_or_path, model_type = ?model_type, core_model_config = ?_core_model_config))]
    async fn load_model(
        &self,
        model_id_or_path: &str,
        model_type: ModelType,
        _core_model_config: Option<CoreModelConfig>, // Not used by diffusion-rs directly for now
    ) -> Result<Arc<dyn Model>> {
        if model_type != ModelType::TextToImage {
            return Err(InferenceError::UnsupportedFeature(
                self.backend_type(),
                format!("Model type {:?} not supported", model_type),
            ));
        }

        // Try to parse as a full preset string first
        if let Some(drs_preset) = string_to_drs_preset(model_id_or_path) {
            info!(preset_str = %model_id_or_path, parsed_preset = ?drs_preset, "Loading model from preset string");
            let mut preset_builder = DiffusionRsPresetBuilder::default();
            preset_builder.preset(drs_preset);
            preset_builder.prompt(""); // Add dummy prompt for preset builder

            match preset_builder.build() {
                Ok((drs_config, drs_model_config)) => {
                    let model_name = format!("preset:{}", model_id_or_path);
                    debug!(name = %model_name, config = ?drs_config, model_config = ?drs_model_config, "Preset built successfully");
                    Ok(Arc::new(DiffusionRsModel {
                        name: model_name,
                        drs_config,
                        drs_model_config,
                    }))
                }
                Err(e) => {
                    error!(preset_str = %model_id_or_path, error = ?e, "Failed to build from preset string");
                    Err(InferenceError::ModelLoad(format!(
                        "Failed to build diffusion-rs preset from string '{}': {:?}",
                        model_id_or_path, e
                    )))
                }
            }
        } else {
            // Treat as a local file path
            info!(path = %model_id_or_path, "Loading model from local file path");
            let mut model_file_path = PathBuf::from(model_id_or_path);

            if !model_file_path.is_absolute() {
                let base_path_str = env::var(DIFFUSION_MODELS_PATH_ENV)
                    .or_else(|_| env::var(DEFAULT_MODELS_PATH_ENV))
                    .map_err(|_| {
                        InferenceError::InvalidConfig(format!(
                            "Neither {} nor {} environment variable is set. Cannot resolve relative model path: {}",
                            DIFFUSION_MODELS_PATH_ENV, DEFAULT_MODELS_PATH_ENV, model_id_or_path
                        ))
                    })?;
                model_file_path = Path::new(&base_path_str).join(model_id_or_path);
            }

            if !model_file_path.exists() {
                return Err(InferenceError::ModelLoad(format!(
                    "Model file not found at resolved path: {}",
                    model_file_path.display()
                )));
            }

            // For local files, we start with a default DiffusionRsApiConfig and primarily configure DiffusionRsModelConfig
            let mut drs_model_config_builder = DiffusionRsModelConfigBuilder::default();
            drs_model_config_builder.model(model_file_path.clone());
            // Users might need to specify VAE, CLIP, T5 paths via CoreModelConfig or extended DiffusionOptions in the future.
            // For now, relying on diffusion-rs to auto-detect or use defaults for these.

            let drs_model_config = drs_model_config_builder.build().map_err(|e| {
                error!(path = %model_file_path.display(), error = ?e, "Failed to build ModelConfig for local file");
                InferenceError::ModelLoad(format!(
                    "Failed to build diffusion-rs ModelConfig for '{}': {:?}",
                    model_id_or_path, e
                ))
            })?;

            // Create a default API config, specific generation params will be set in `generate_image`
            let drs_config = DiffusionRsApiConfigBuilder::default()
                .prompt("") // Prompt is set at generation time
                .build()
                .map_err(|e| InferenceError::InternalError(format!("Failed to build default api_config: {:?}",e)))?;


            debug!(name = %model_id_or_path, model_config = ?drs_model_config, "Local model configured");
            Ok(Arc::new(DiffusionRsModel {
                name: model_id_or_path.to_string(),
                drs_config, // Default API config
                drs_model_config,
            }))
        }
    }

    async fn list_available_models(&self) -> Result<Vec<String>> {
        let mut presets = Vec::new();

        // Presets without WeightType argument
        presets.push(drs_preset_to_string(DiffusionRsPreset::StableDiffusion1_4));
        presets.push(drs_preset_to_string(DiffusionRsPreset::StableDiffusion1_5));
        presets.push(drs_preset_to_string(DiffusionRsPreset::StableDiffusion2_1));
        presets.push(drs_preset_to_string(DiffusionRsPreset::StableDiffusion3MediumFp16));
        presets.push(drs_preset_to_string(DiffusionRsPreset::SDXLBase1_0));
        presets.push(drs_preset_to_string(DiffusionRsPreset::SDTurbo));
        presets.push(drs_preset_to_string(DiffusionRsPreset::SDXLTurbo1_0Fp16));
        presets.push(drs_preset_to_string(DiffusionRsPreset::StableDiffusion3_5MediumFp16));
        presets.push(drs_preset_to_string(DiffusionRsPreset::StableDiffusion3_5LargeFp16));
        presets.push(drs_preset_to_string(DiffusionRsPreset::StableDiffusion3_5LargeTurboFp16));
        presets.push(drs_preset_to_string(DiffusionRsPreset::JuggernautXL11));
        
        // Common weight types to generate strings for
        let common_weight_types = vec![
            DiffusionRsWeightType::SD_TYPE_F32,
            DiffusionRsWeightType::SD_TYPE_F16,
            DiffusionRsWeightType::SD_TYPE_Q4_0,
            DiffusionRsWeightType::SD_TYPE_Q4_K,
            DiffusionRsWeightType::SD_TYPE_Q8_0,
            // Add other relevant/common weight types here if needed
        ];

        // Presets with WeightType argument
        for wt in &common_weight_types {
            presets.push(drs_preset_to_string(DiffusionRsPreset::Flux1Dev(*wt)));
            presets.push(drs_preset_to_string(DiffusionRsPreset::Flux1Schnell(*wt)));
            presets.push(drs_preset_to_string(DiffusionRsPreset::Flux1Mini(*wt)));
        }
        
        // Deduplicate and sort for consistent output, though order might not be strictly necessary
        presets.sort();
        presets.dedup();

        Ok(presets)
    }
}

pub struct DiffusionRsModel {
    name: String,
    drs_config: DiffusionRsApiConfig, // Base config from load time (e.g., from preset)
    drs_model_config: DiffusionRsModelConfig, // Model weights, device info
}

// UNSAFE: Assuming DiffusionRsModelConfig can be safely sent and shared.
// This is because diffusion-rs operations like txt2img are synchronous and likely
// perform all their work on the calling thread, or manage their own internal synchronization
// if they use threads. This requires careful verification with the diffusion-rs library's guarantees.
unsafe impl Send for DiffusionRsModel {}
unsafe impl Sync for DiffusionRsModel {}

impl Model for DiffusionRsModel {
    fn model_type(&self) -> ModelType {
        ModelType::TextToImage
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn as_text_to_image(&self) -> Option<&dyn TextToImageModel> {
        Some(self)
    }
}

#[async_trait]
impl TextToImageModel for DiffusionRsModel {
    #[instrument(skip(self, prompt, options), fields(model_name = %self.name, prompt_len = prompt.len()))]
    async fn generate_image(
        &self,
        prompt: &str,
        options: Option<DiffusionOptions>,
    ) -> Result<ImageOutput> {
        // Start by creating a builder from the model's base API config
        let mut drs_api_config_builder = DiffusionRsApiConfigBuilder::from(self.drs_config.clone());

        // Always set the current prompt
        drs_api_config_builder.prompt(prompt.to_string());

        let mut output_format = ImageOutputFormat::Png; // Default to PNG
        let mut final_output_path_for_result: Option<PathBuf> = None;

        if let Some(opts) = options {
            if let Some(neg_prompt) = opts.negative_prompt {
                drs_api_config_builder.negative_prompt(neg_prompt);
            }
            if let Some(w) = opts.width {
                drs_api_config_builder.width(w as i32); 
            }
            if let Some(h) = opts.height {
                drs_api_config_builder.height(h as i32); 
            }
            if let Some(s) = opts.steps {
                drs_api_config_builder.steps(s as i32); 
            }
            if let Some(cfg_s) = opts.cfg_scale {
                drs_api_config_builder.cfg_scale(cfg_s);
            }
            if let Some(sampler_kind) = opts.sampler {
                if let Some(drs_sampler) = map_to_diffusion_rs_sampler(Some(sampler_kind)) {
                    drs_api_config_builder.sampling_method(drs_sampler);
                }
            }
            if let Some(s) = opts.seed {
                drs_api_config_builder.seed(s as i64); 
            }
            if let Some(path) = opts.output_path {
                // User specified an output path in options
                final_output_path_for_result = Some(path.clone());
                drs_api_config_builder.output(path); 
            }
            if let Some(format) = opts.output_format {
                output_format = format;
                if format != ImageOutputFormat::Png {
                    error!("Diffusion-rs backend currently only supports PNG output directly. Requested: {:?}", format);
                }
            }
        }
        
        // If no output path was provided in options, generate a temporary one.
        if final_output_path_for_result.is_none() {
            let temp_dir = env::temp_dir();
            let file_name = format!(
                "diffusion_rs_output_{}.png",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            );
            let temp_file_path = temp_dir.join(file_name);
            final_output_path_for_result = Some(temp_file_path.clone());
            drs_api_config_builder.output(temp_file_path);
        }

        let mut current_drs_config = drs_api_config_builder.build().map_err(|e| {
            InferenceError::InvalidConfig(format!("Failed to build diffusion-rs API config: {:?}", e))
        })?;

        info!(config = ?current_drs_config, model_config = ?self.drs_model_config, "Generating image with diffusion-rs");

        let mut model_config_to_pass = self.drs_model_config.clone();
        
        // Ensure the directory for the output path exists.
        // Use the path we *intend* to use, which is final_output_path_for_result.
        if let Some(out_path) = &final_output_path_for_result { 
            if let Some(parent_dir) = out_path.parent() {
                if !parent_dir.exists() {
                    std::fs::create_dir_all(parent_dir).map_err(|e| {
                        InferenceError::BackendError(anyhow::anyhow!(
                            "Failed to create output directory {}: {}",
                            parent_dir.display(),
                            e
                        ))
                    })?;
                    info!("Created output directory: {}", parent_dir.display());
                }
            }
        } else {
            // This case should ideally not be reached if logic above is correct,
            // as final_output_path_for_result should always be Some by now.
            return Err(InferenceError::InternalError("Output path was not determined before directory creation check.".to_string()));
        }

        txt2img(&mut current_drs_config, &mut model_config_to_pass).map_err(|e| {
            error!(error = ?e, "txt2img generation failed");
            InferenceError::GenerationError(format!("diffusion-rs txt2img failed: {:?}", e))
        })?;

        // Use the output path we decided on and passed to the builder.
        let confirmed_output_path = final_output_path_for_result.ok_or_else(|| {
            InferenceError::InternalError("Output path was unexpectedly None after generation.".to_string())
        })?;

        info!(path = %confirmed_output_path.display(), "Image generation successful");

        match output_format {
            ImageOutputFormat::Png => {
                Ok(ImageOutput::File(confirmed_output_path))
            }
            _ => {
                 Ok(ImageOutput::File(confirmed_output_path))
            }
        }
    }
} 