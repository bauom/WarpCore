use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Specifies the type of inference backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BackendType {
    LlamaCpp,
    OpenAI,
    Anthropic,
    DiffusionRs,
    Hive, // Candle, // Planned
          // Groq, // Planned
          // Modal, // Planned
}

/// Specifies the type of model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ModelType {
    TextToText,
    TextToImage,
    // ImageToText, // Planned
    // TextToImage, // Planned
}

/// General configuration applicable to any backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)] // Allows either Local or Api directly
pub enum BackendConfig {
    Local(LocalBackendConfig),
    Api(ApiConfig),
}

/// Configuration specific to local inference backends (e.g., llama.cpp, candle).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LocalBackendConfig {
    pub threads: Option<u32>,
    pub use_gpu: Option<bool>,
    pub gpu_layers: Option<u32>,
    pub llama_cpp: Option<LlamaCppSpecificConfig>,
    // Add other common local options: mmap, etc.
}

/// Configuration specific to the Llama.cpp backend.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LlamaCppSpecificConfig {
    pub use_mmap: Option<bool>,
    pub use_mlock: Option<bool>,
    pub numa_strategy: Option<String>, // Represent NumaStrategy:: variants as strings for now
}

impl LlamaCppSpecificConfig {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_mmap(mut self, use_mmap: bool) -> Self {
        self.use_mmap = Some(use_mmap);
        self
    }

    pub fn with_mlock(mut self, use_mlock: bool) -> Self {
        self.use_mlock = Some(use_mlock);
        self
    }

    pub fn with_numa_strategy(mut self, strategy: impl Into<String>) -> Self {
        self.numa_strategy = Some(strategy.into());
        self
    }
}

impl LocalBackendConfig {
    pub fn new() -> Self {
        Default::default()
    }
    // Builder methods...
    pub fn with_threads(mut self, threads: u32) -> Self {
        self.threads = Some(threads);
        self
    }
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = Some(use_gpu);
        self
    }
    pub fn with_gpu_layers(mut self, layers: u32) -> Self {
        self.gpu_layers = Some(layers);
        self
    }
    pub fn with_llama_cpp_config(mut self, config: LlamaCppSpecificConfig) -> Self {
        self.llama_cpp = Some(config);
        self
    }
}

/// Configuration specific to API-based inference backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub api_key: Option<String>, // Can be loaded from env
    pub organization: Option<String>,
    pub base_url: Option<String>,
    pub timeout: Option<Duration>,
    // Add other common API options
}

impl ApiConfig {
    /// Creates a new ApiConfig, optionally providing an API key directly.
    /// If `None`, the backend implementation should attempt to load from environment variables.
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key,
            organization: None,
            base_url: None,
            timeout: None,
        }
    }
    // Builder methods...
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout = Some(Duration::from_secs(timeout_secs));
        self
    }
}

/// Configuration specific to a model being loaded.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelConfig {
    pub context_size: Option<u32>,
    // Add seed, lora adapters etc.
}

impl ModelConfig {
    pub fn new() -> Self {
        Default::default()
    }
    // Builder methods...
    pub fn with_context_size(mut self, size: u32) -> Self {
        self.context_size = Some(size);
        self
    }
}

/// Options controlling the text generation process.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationOptions {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub stop_sequences: Option<Vec<String>>,
    pub system: Option<String>,
    // Add seed, repetition penalty etc.
}

impl GenerationOptions {
    pub fn new() -> Self {
        Default::default()
    }
    // Builder methods...
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }
    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }
    pub fn with_stop_sequences(mut self, stop: Vec<String>) -> Self {
        self.stop_sequences = Some(stop);
        self
    }
    pub fn with_system(mut self, system_prompt: String) -> Self {
        self.system = Some(system_prompt);
        self
    }
}

/// Options specific to image generation models (e.g., diffusion models).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiffusionOptions {
    pub negative_prompt: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub steps: Option<u32>,
    pub cfg_scale: Option<f32>,
    pub sampler: Option<SamplerKind>,
    pub seed: Option<u64>,
    pub output_path: Option<PathBuf>,
    pub output_format: Option<ImageOutputFormat>,
    // Add other diffusion-specific options as needed
}

impl DiffusionOptions {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_negative_prompt(mut self, prompt: String) -> Self {
        self.negative_prompt = Some(prompt);
        self
    }

    pub fn with_width(mut self, width: u32) -> Self {
        self.width = Some(width);
        self
    }

    pub fn with_height(mut self, height: u32) -> Self {
        self.height = Some(height);
        self
    }

    pub fn with_steps(mut self, steps: u32) -> Self {
        self.steps = Some(steps);
        self
    }

    pub fn with_cfg_scale(mut self, scale: f32) -> Self {
        self.cfg_scale = Some(scale);
        self
    }

    pub fn with_sampler(mut self, sampler: SamplerKind) -> Self {
        self.sampler = Some(sampler);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_output_path(mut self, path: PathBuf) -> Self {
        self.output_path = Some(path);
        self
    }

    pub fn with_output_format(mut self, format: ImageOutputFormat) -> Self {
        self.output_format = Some(format);
        self
    }
}

/// Specifies the sampling method for diffusion models.
/// This should mirror options available in the `diffusion-rs` crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SamplerKind {
    EulerA,       // EULER_A
    Euler,        // EULER
    Heun,         // HEUN
    Dpm2,         // DPM2
    Dpmpp2sA,     // DPMPP2S_A
    Dpmpp2m,      // DPMPP2M
    Dpmpp2mv2,    // DPMPP2Mv2 - NEW
    Ipndm,        // IPNDM - NEW (was Pndm)
    IpndmV,       // IPNDM_V - NEW
    Lcm,          // LCM - NEW
    DdimTrailing, // DDIM_TRAILING - NEW (was Ddim)
    Tcd,          // TCD - NEW
                  // LMS and UniPc seem to be missing from diffusion-rs 0.1.9 sample_method_t
                  // Add others as supported by diffusion-rs
}

/// Specifies the desired output format for generated images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImageOutputFormat {
    Png,
    Jpg,
    // Add other formats if needed
}
