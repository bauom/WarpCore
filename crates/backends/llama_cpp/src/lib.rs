use async_trait::async_trait;
use async_stream::try_stream;
use futures::Stream;
use warpcore_core::config::{
    BackendConfig, BackendType, GenerationOptions, LocalBackendConfig,
    ModelConfig, ModelType,
};
use warpcore_core::error::{InferenceError, Result};
use warpcore_core::traits::{InferenceService, Model, TextToTextModel};
use llama_cpp_2::{
    context::{params::LlamaContextParams},
    llama_backend::{LlamaBackend, NumaStrategy},
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    token::LlamaToken,
};
use once_cell::sync::OnceCell;
use std::{
    env,
    fs,
    num::NonZeroU32,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{mpsc, Arc},
    thread,
};
use tracing::{debug, error, info, instrument, warn};

// Global LlamaBackend instance, initialized once.
static LLAMA_BACKEND: OnceCell<LlamaBackend> = OnceCell::new();

// Custom error mapping for llama_cpp_2 errors
fn map_llama_error(err: llama_cpp_2::LLamaCppError) -> InferenceError {
    match err {
        llama_cpp_2::LLamaCppError::LlamaModelLoadError(e) => {
            InferenceError::ModelLoad(format!("Llama model load error: {}", e))
        }
        llama_cpp_2::LLamaCppError::LlamaContextLoadError(e) => {
            InferenceError::BackendError(anyhow::anyhow!("Llama context load error: {}", e))
        }
        llama_cpp_2::LLamaCppError::BackendAlreadyInitialized => {
            // This should ideally not happen if OnceCell is used correctly, but map it just in case.
            InferenceError::BackendError(anyhow::anyhow!("Llama backend already initialized"))
        }
        llama_cpp_2::LLamaCppError::DecodeError(e) => {
            InferenceError::GenerationError(format!("Llama decode error: {}", e))
        }
        _ => InferenceError::BackendError(anyhow::anyhow!("Llama backend error: {}", err)),
    }
}

/// Inference service implementation for llama.cpp.
#[derive(Debug, Clone)]
pub struct LlamaCppService {
    config: Option<LocalBackendConfig>, // Store general local config if needed later
}

impl LlamaCppService {
    /// Creates a new LlamaCppService.
    /// Initializes the global LlamaBackend if not already done.
    pub fn new(config: Option<BackendConfig>) -> Result<Self> {
        let local_config = match config {
            Some(BackendConfig::Local(local_cfg)) => Some(local_cfg),
            Some(BackendConfig::Api(_)) => {
                return Err(InferenceError::InvalidConfig(
                    "LlamaCpp backend requires LocalBackendConfig, not ApiConfig".to_string(),
                ))
            }
            None => None, // Use defaults
        };

        // Extract specific llama_cpp config from local_config
        let specific_config_option = local_config.as_ref().and_then(|lc| lc.llama_cpp.clone());

        // Need to initialize backend
        let _ = LLAMA_BACKEND.get_or_try_init(|| {
            // Get NUMA strategy
            let numa_strategy = match &specific_config_option {
                Some(cfg) => {
                    match cfg.numa_strategy.as_deref() {
                        Some("DISABLED") => Some(NumaStrategy::DISABLED),
                        Some("NUMACTL") => Some(NumaStrategy::NUMACTL),
                        Some("DISTRIBUTE") => Some(NumaStrategy::DISTRIBUTE),
                        Some("MIRROR") => Some(NumaStrategy::MIRROR),
                        Some(other) => {
                            return Err(InferenceError::InvalidConfig(format!(
                                "Invalid or unsupported Llama.cpp NUMA strategy: {}",
                                other
                            )))
                        }
                        None => None,
                    }
                }
                None => None,
            };

            // Initialize backend with appropriate strategy
            let backend_result = if let Some(strategy) = numa_strategy {
                LlamaBackend::init_numa(strategy)
            } else {
                LlamaBackend::init()
            };

            backend_result.map_err(|e| {
                if let llama_cpp_2::LLamaCppError::BackendAlreadyInitialized = e {
                    InferenceError::BackendError(anyhow::anyhow!(
                        "Backend initialization race condition detected"
                    ))
                } else {
                    map_llama_error(e)
                }
            }).map(|mut backend| {
                backend.void_logs();
                backend
            })
        })
        .map_err(|e| {
            tracing::error!(error = ?e, "Failed to initialize LlamaBackend");
            e
        })?;

        Ok(Self {
            config: local_config,
        })
    }

    
}

/// Represents a loaded Llama.cpp model.
/// Context is created per-request in generate/generate_stream.
pub struct LlamaCppModel {
    name: String, // Model path
    model: Arc<LlamaModel>,
    config: Option<ModelConfig>,
    backend_config: Option<LocalBackendConfig>,
}

// Helper to create context parameters
fn create_context_params(
    backend_config: &LocalBackendConfig,
    model_config: &ModelConfig,
) -> Result<LlamaContextParams> {
    let mut params = LlamaContextParams::default();

    // Use configured or default context size
    let ctx_size = model_config.context_size.unwrap_or(2048);
    params = params.with_n_ctx(Some(NonZeroU32::new(ctx_size).ok_or_else(|| 
        InferenceError::InvalidConfig("Context size cannot be zero".to_string()))?));
    debug!(context_size = ctx_size, "Setting context size");

    // Use configured or default thread count
    let threads = backend_config.threads.unwrap_or_else(|| {
        let count = num_cpus::get() as u32;
        debug!(threads = count, "Defaulting threads to num_cpus");
        count
    });
    params = params.with_n_threads(threads as i32);
    params = params.with_n_threads_batch(threads as i32);
    debug!(threads = threads, "Setting threads");

    // mmap and mlock are model params, removed from here

    Ok(params)
}

// --- Trait Implementations (to be filled) ---

#[async_trait]
impl InferenceService for LlamaCppService {
    fn backend_type(&self) -> BackendType {
        BackendType::LlamaCpp
    }

    fn supported_model_types(&self) -> &[ModelType] {
        &[ModelType::TextToText] // Currently only support TextToText
    }

    #[instrument(skip(self, config), fields(model=%model_id, type=?model_type))]
    async fn load_model(
        &self,
        model_id: &str,
        model_type: ModelType,
        config: Option<ModelConfig>,
    ) -> Result<Arc<dyn Model>> {
        if model_type != ModelType::TextToText {
            return Err(InferenceError::UnsupportedFeature(
                self.backend_type(),
                format!("Model type {:?} not supported", model_type),
            ));
        }

        // Resolve model path using MODELS_PATH
        let base_path_var = "MODELS_PATH";
        let model_file_path = match env::var(base_path_var) {
            Ok(base_path) => {
                let relative_path = model_id.replace(':', "/");
                let full_path = Path::new(&base_path).join(relative_path);
                if full_path.exists() {
                    full_path.to_string_lossy().to_string()
                } else {
                    return Err(InferenceError::ModelLoad(format!(
                        "Model file derived from ID '{}' not found at expected path: {}",
                        model_id,
                        full_path.display()
                    )));
                }
            }
            Err(_) => {
                return Err(InferenceError::InvalidConfig(format!(
                    "{} environment variable not set. Cannot resolve model path for ID '{}'.",
                    base_path_var,
                    model_id
                )));
            }
        };
        // Now model_file_path contains the actual path to the .gguf file

        let backend = LLAMA_BACKEND
            .get()
            .ok_or_else(|| {
                InferenceError::BackendError(anyhow::anyhow!("Llama backend not initialized"))
            })?;

        let local_backend_config = self.config.clone().unwrap_or_default();
        let specific_config = local_backend_config.llama_cpp.as_ref(); // Needed for mmap/mlock

        debug!(?local_backend_config, ?config, "Loading Llama.cpp model");

        let model_params = {
            let mut params = LlamaModelParams::default(); // Make mutable
            
            // Set GPU layers using setter
            if let Some(layers) = local_backend_config.gpu_layers {
                params = params.with_n_gpu_layers(layers);
                debug!(gpu_layers = layers, "Setting GPU layers");
            } else {
                params = params.with_n_gpu_layers(0); // Default to 0
                debug!("Defaulting to 0 GPU layers");
            }

            // Set mmap/mlock using setters exactly as per docs.rs/0.1.92 - COMMENTED OUT DUE TO LINTER ERRORS
            let use_mmap = specific_config.and_then(|c| c.use_mmap).unwrap_or(true);
            // params = params.with_mmap(use_mmap); // Use with_mmap
            debug!(use_mmap = use_mmap, "Setting mmap usage (currently disabled)");

            let use_mlock = specific_config.and_then(|c| c.use_mlock).unwrap_or(false);
            // params = params.with_mlock(use_mlock); // Use with_mlock
            debug!(use_mlock = use_mlock, "Setting mlock usage (currently disabled)");

            params // Return configured params
        };

        debug!("Loading model file...");
        let llama_model = Arc::new(
            LlamaModel::load_from_file(backend, &model_file_path, &model_params)
                .map_err(|e| InferenceError::ModelLoad(format!("Failed to load model: {}", e)))?,
        );
        debug!("Model file loaded successfully");

        // Context is NOT created here anymore

        let model_instance = LlamaCppModel {
            name: model_id.to_string(),
            model: llama_model,
            config,
            backend_config: self.config.clone(),
        };

        // This is now valid as LlamaCppModel no longer has a lifetime
        Ok(Arc::new(model_instance))
    }

    async fn list_available_models(&self) -> Result<Vec<String>> {
        let base_path_var = "MODELS_PATH";
        let base_path_str = match env::var(base_path_var) {
            Ok(path) => path,
            Err(_) => {
                let err_msg = format!("{} environment variable not set. Cannot list local models.", base_path_var);
                error!(err_msg);
                return Err(InferenceError::InvalidConfig(err_msg)); 
            }
        };

        let base_path = PathBuf::from(&base_path_str);
        if !base_path.is_dir() {
            let err_msg = format!(
                "{} env var points to invalid path: {}", 
                base_path_var, 
                base_path.display()
            );
            error!(path = %base_path.display(), "{} is not a valid directory.", base_path_var);
            return Err(InferenceError::InvalidConfig(err_msg));
        }

        let mut model_ids = Vec::new();
        let mut dirs_to_visit = vec![base_path.clone()];

        info!(path = %base_path.display(), "Scanning for .gguf models...");

        while let Some(dir) = dirs_to_visit.pop() {
            match fs::read_dir(&dir) {
                Ok(entries) => {
                    for entry_result in entries {
                        match entry_result {
                            Ok(entry) => {
                                let path = entry.path();
                                debug!(?path, "Scanning path");
                                if path.is_dir() {
                                    debug!(?path, "Adding directory to visit list");
                                    dirs_to_visit.push(path);
                                } else if path.is_file() {
                                    debug!(?path, "Found file");
                                    let extension = path.extension()
                                                        .and_then(|os_str| os_str.to_str())
                                                        .map(|s| s.to_lowercase());
                                    debug!(?extension, "File extension");
                                    if extension == Some("gguf".to_string()) {
                                        info!(path = %path.display(), "Found potential .gguf file, attempting to process."); 
                                        match path.strip_prefix(&base_path) {
                                            Ok(relative_path) => {
                                                info!(relative = ?relative_path, base = ?base_path, "Successfully stripped prefix."); 
                                                
                                                // Reconstruct ID assuming provider/user/repo/filename structure
                                                let components: Vec<_> = relative_path.components().map(|comp| comp.as_os_str().to_string_lossy()).collect();
                                                
                                                if components.len() == 4 {
                                                    let provider = &components[0];
                                                    let user_repo = &components[1..3].join("/"); // Join user/repo with slash
                                                    let filename = &components[3];
                                                    
                                                    let model_id = format!("{}:{}:{}", provider, user_repo, filename);
                                                    
                                                    info!(id = %model_id, path = %path.display(), "Found and added model (valid structure).");
                                                    model_ids.push(model_id);
                                                } else {
                                                    debug!(path = %path.display(), components = ?components, "Skipping model (invalid structure - expected 4 components: provider/user/repo/filename)");
                                                }
                                            },
                                            Err(e) => {
                                                error!(root = %base_path.display(), file = %path.display(), error = ?e, "Failed to strip base path prefix");
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                warn!(path = %dir.display(), error = ?e, "Failed to read directory entry");
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!(path = %dir.display(), error = ?e, "Failed to read directory contents");
                }
            }
        }

        info!(count = model_ids.len(), "Finished scanning for models.");
        Ok(model_ids)
    }
    // list_available_models remains default (empty Ok)
}

impl Model for LlamaCppModel {
    fn model_type(&self) -> ModelType {
        ModelType::TextToText
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn as_text_to_text(&self) -> Option<&dyn TextToTextModel> {
        Some(self)
    }
}

// No lifetime needed
#[async_trait]
impl TextToTextModel for LlamaCppModel {
    #[instrument(skip(self, prompt, options), fields(model=%self.name()))]
    async fn generate(
        &self,
        prompt: &str,
        options: Option<GenerationOptions>,
    ) -> Result<String> {
        let gen_options = options.clone().unwrap_or_default();
        let max_tokens = gen_options.max_tokens.unwrap_or(512);

        // Get backend instance
        let backend = LLAMA_BACKEND
            .get()
            .ok_or_else(|| InferenceError::BackendError(anyhow::anyhow!("Llama backend not initialized")))?;

        // Prepare context parameters
        let model_config = self.config.clone().unwrap_or_default();
        let backend_config = self.backend_config.clone().unwrap_or_default();
        let context_params = create_context_params(&backend_config, &model_config)?;

        // Create context for this request
        let mut context = self
            .model
            .new_context(backend, context_params)
            .map_err(|e| InferenceError::BackendError(anyhow::anyhow!("Failed to create context: {}", e)))?;
        
        // Tokenize prompt using model.str_to_token
        let tokens: Vec<LlamaToken> = self
            .model
            .str_to_token(prompt, AddBos::Always) // Use str_to_token and AddBos enum
            .map_err(|e| InferenceError::GenerationError(format!("Tokenization failed: {}", e)))?;
        
        debug!(prompt_tokens = tokens.len(), "Tokenized prompt");

        let n_ctx = context.n_ctx() as usize;
        let mut batch = LlamaBatch::new(n_ctx, 1);

        // Add prompt tokens to batch & decode
        batch.add_sequence(&tokens, 0, true).map_err(map_batch_add_error)?;
        context.decode(&mut batch).map_err(map_decode_error)?;
        
        let mut decoded_text = String::new();
        let mut generated_tokens_count = 0;

        // Generation loop - simplified to match available API
        while generated_tokens_count < max_tokens {
            let candidates = context.candidates_ith(batch.n_tokens() - 1).collect::<Vec<_>>();
            
            // Basic sampling - use logit() and id() from LlamaTokenData
            let next_token = if candidates.is_empty() {
                break;
            } else {
                candidates.into_iter().max_by(|a, b| a.logit().total_cmp(&b.logit())).unwrap().id()
            };

            // Check for EOS token using model.token_eos()
            if next_token == self.model.token_eos() { 
                break; 
            }

            // Convert token to string using model.token_to_str
            let token_str = self.model.token_to_str(next_token, Special::Tokenize)
                .map_err(|e| InferenceError::GenerationError(format!("Token decoding failed: {}", e)))?;
            
            decoded_text.push_str(&token_str);
            generated_tokens_count += 1;

            if check_stop_sequences(&decoded_text, &gen_options) { 
                break; 
            }

            batch.clear();
            batch.add(next_token, batch.n_tokens(), &[0], true).map_err(map_batch_add_error)?;
            context.decode(&mut batch).map_err(map_decode_error)?;
            debug!(iteration=generated_tokens_count, token=?next_token, text=%token_str, "Generated token");
        }
        
        debug!(total_tokens=generated_tokens_count, "Generation loop finished");
        Ok(decoded_text)
    }

    #[instrument(skip(self, prompt, options), fields(model=%self.name()))]
    fn generate_stream<'a>(
        &'a self,
        prompt: &'a str,
        options: Option<GenerationOptions>,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        let gen_options = options.unwrap_or_default();
        let max_tokens = gen_options.max_tokens.unwrap_or(512);
        let prompt_string = prompt.to_string(); // Clone prompt for the thread

        // Clone required Arcs and configs for the thread
        let model_arc = Arc::clone(&self.model);
        let model_config_clone = self.config.clone().unwrap_or_default();
        let backend_config_clone = self.backend_config.clone().unwrap_or_default();

        info!("Setting up generate_stream thread...");

        Box::pin(try_stream! {
            // Channel to receive tokens from the generation thread
            let (tx, rx) = mpsc::channel::<Result<String>>();

            info!("Spawning generation thread...");
            let generation_handle = thread::spawn(move || {
                info!("Generation thread started.");
                // This block runs in a separate thread
                let backend = match LLAMA_BACKEND.get() {
                    Some(b) => b,
                    None => {
                        error!("Backend not initialized in generation thread");
                        let _ = tx.send(Err(InferenceError::BackendError(anyhow::anyhow!(
                            "Llama backend not initialized in generation thread"
                        ))));
                        return;
                    }
                };
                info!("Backend obtained in thread.");

                let context_params = match create_context_params(&backend_config_clone, &model_config_clone) {
                     Ok(p) => p,
                     Err(e) => { 
                         error!(error = ?e, "Failed to create context params in thread");
                         let _ = tx.send(Err(e)); 
                         return; 
                     }
                };
                info!("Context params created in thread.");

                let mut context = match model_arc.new_context(backend, context_params) {
                    Ok(c) => c,
                    Err(e) => { 
                        error!(error = ?e, "Failed to create context in thread");
                        let _ = tx.send(Err(InferenceError::BackendError(anyhow::anyhow!("Failed to create context: {}", e)))); 
                        return; 
                    }
                };
                info!("Context created in thread.");

                let tokens: Vec<LlamaToken> = match model_arc.str_to_token(&prompt_string, AddBos::Always) {
                    Ok(t) => t,
                    Err(e) => { 
                        error!(error = ?e, "Tokenization failed in thread");
                        let _ = tx.send(Err(InferenceError::GenerationError(format!("Tokenization failed: {}", e)))); 
                        return; 
                    }
                };
                info!(count = tokens.len(), "Tokenization complete in thread.");

                let n_ctx = context.n_ctx() as usize;
                let mut batch = LlamaBatch::new(n_ctx, 1);
                
                match batch.add_sequence(&tokens, 0, true) {
                    Ok(_) => { info!("Added initial sequence to batch in thread."); },
                    Err(e) => { 
                        error!(error = ?e, "Failed to add sequence to batch in thread");
                        let _ = tx.send(Err(map_batch_add_error(e))); 
                        return; 
                    }
                }
                
                match context.decode(&mut batch) {
                    Ok(_) => { info!("Decoded initial batch in thread."); },
                    Err(e) => { 
                        error!(error = ?e, "Failed to decode initial batch in thread");
                        let _ = tx.send(Err(map_decode_error(e))); 
                        return; 
                    }
                }
                
                let mut generated_tokens_count = 0u32;
                let mut current_text = String::new();

                info!(max_tokens, "Starting stream generation loop in thread...");
                while generated_tokens_count < max_tokens {
                    debug!(iteration = generated_tokens_count, "Generation loop iteration start.");
                    
                    // --- Getting Candidates ---
                    debug!("Getting candidates...");
                    let candidates = context.candidates_ith(batch.n_tokens() - 1).collect::<Vec<_>>();
                    debug!(count = candidates.len(), "Got candidates.");
                    
                    // --- Sampling ---
                    debug!("Sampling next token...");
                    let next_token = if candidates.is_empty() {
                        warn!("No candidates found, breaking generation loop.");
                        break;
                    } else {
                        candidates.into_iter().max_by(|a, b| a.logit().total_cmp(&b.logit())).unwrap().id()
                    };
                    // Use Debug format for LlamaToken in logging
                    debug!(token = ?next_token, "Sampled token.");

                    // --- EOS Check ---
                    debug!("Checking for EOS...");
                    if model_arc.token_eos() == next_token { 
                        info!("EOS token detected, breaking generation loop.");
                        break; 
                    }
                    debug!("Not EOS.");

                    // --- Token to String ---
                    debug!("Converting token to string...");
                    let token_str = match model_arc.token_to_str(next_token, Special::Tokenize) {
                        Ok(s) => s,
                        Err(e) => { 
                            error!(error = ?e, "Token decoding failed in thread");
                            let _ = tx.send(Err(InferenceError::GenerationError(format!("Token decoding failed: {}", e)))); 
                            return; // Exit thread on error
                        }
                    };
                    debug!(text = %token_str, "Converted token to string.");

                    // --- Sending Token ---
                    debug!("Sending token to stream...");
                    if tx.send(Ok(token_str.clone())).is_err() {
                        // Receiver disconnected, stop generation
                        error!("Stream receiver disconnected during send. Stopping thread.");
                        return; // Exit thread
                    }
                    debug!("Token sent.");
                    
                    current_text.push_str(&token_str);
                    generated_tokens_count += 1;

                    // --- Stop Sequence Check ---
                    debug!("Checking stop sequences...");
                    if check_stop_sequences(&current_text, &gen_options) { 
                        info!("Stop sequence detected, breaking generation loop.");
                        break; 
                    }
                    debug!("No stop sequence.");

                    // --- Update Batch ---
                    debug!("Updating batch...");
                    batch.clear();
                    match batch.add(next_token, batch.n_tokens(), &[0], true) {
                        Ok(_) => {},
                        Err(e) => { 
                            error!(error = ?e, "Failed to add token to batch in thread");
                            let _ = tx.send(Err(map_batch_add_error(e))); 
                            return; // Exit thread
                        }
                    }
                    debug!("Batch updated.");
                    
                    // --- Decode Batch ---
                    debug!("Decoding batch...");
                    match context.decode(&mut batch) {
                        Ok(_) => {},
                        Err(e) => { 
                            error!(error = ?e, "Failed to decode batch in thread");
                            let _ = tx.send(Err(map_decode_error(e))); 
                            return; // Exit thread
                        }
                    }
                    debug!(iteration=generated_tokens_count, "Generation loop iteration end.");
                }
                
                info!(total_tokens=generated_tokens_count, reason=if generated_tokens_count == max_tokens { "max_tokens reached" } else { "EOS or stop sequence" }, "Stream generation loop in thread finished.");
                // Dropping tx signals the end of the stream
                info!("Generation thread finished and dropping sender channel.");
            });

            info!("Starting receiver loop...");
            // This part runs in the async context
            while let Ok(token_result) = rx.recv() {
                match token_result {
                    Ok(token) => {
                        debug!(token = %token, "Received token from thread.");
                        yield token
                    },
                    Err(e) => {
                        error!(error = ?e, "Received error from thread.");
                        // Propagate the error yielded by the thread
                        Err(e)?;
                    }
                }
            }
            info!("Receiver loop finished (channel closed).");

            info!("Waiting for generation thread to join...");
            // Ensure the thread finishes
            if let Err(e) = generation_handle.join() {
                error!(error=?e, "Generation thread panicked");
                Err(InferenceError::BackendError(anyhow::anyhow!("Generation thread panicked")))?;
            }
            info!("Generation thread joined successfully.");
        })
    }
}

// --- Helper functions for error mapping and stop sequences ---

fn map_batch_add_error(e: llama_cpp_2::llama_batch::BatchAddError) -> InferenceError {
    InferenceError::GenerationError(format!("Failed to add token to batch: {}", e))
}

fn map_decode_error(e: llama_cpp_2::DecodeError) -> InferenceError {
    InferenceError::GenerationError(format!("Decoding failed: {}", e))
}

fn check_stop_sequences(current_text: &str, options: &GenerationOptions) -> bool {
    if let Some(stop_seqs) = &options.stop_sequences {
        if stop_seqs.iter().any(|stop| current_text.ends_with(stop)) {
            debug!("Stop sequence reached");
            return true;
        }
    }
    false
} 