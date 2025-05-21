# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Core support for Text-to-Image:
  - `ModelType::TextToImage` enum variant in `crates/core/src/config.rs`.
  - `TextToImageModel` trait in `crates/core/src/traits.rs`.
  - `DiffusionOptions`, `SamplerKind`, `ImageOutputFormat`, `ImageOutput` structs/enums in `crates/core/src/config.rs`.
  - `as_text_to_image()` method to the base `Model` trait in `crates/core/src/traits.rs`.
- `diffusion-rs` backend (`inference-lib-diffusion-rs` crate):
  - Implements `InferenceService` and `TextToImageModel`.
  - Uses `diffusion-rs` crate (version `0.1.9`).
  - Supports loading models by preset name (dynamically mapped from all `diffusion_rs::preset::Preset` variants, including those with `WeightType`) or local file path (resolved via `$DIFFUSION_MODELS_PATH` or `$MODELS_PATH`, supports `.safetensors`, `.ckpt`, `.gguf`).
  - `generate_image` method implemented using `diffusion_rs::api::txt2img`.
  - Handles mapping of `SamplerKind` (from `inference-lib-core`) to `diffusion_rs::api::SampleMethod`.
  - Includes a `README.md` for the backend detailing usage, caveats, and TODOs.
- Integration tests for `diffusion-rs` backend (`crates/examples/tests/diffusion_rs_integration.rs`):
  - Tests preset generation (e.g., `SDXLTurbo1_0Fp16`) and local model loading.
  - Includes `setup()` function for `tracing_subscriber` and `dotenv` initialization.
- Initial implementation of the `llama_cpp` backend (`inference-lib-llama-cpp` crate).
  - Uses `llama-cpp-2` crate (version `0.1.92`) as the underlying engine.
  - Implements `InferenceService` for loading models (`LlamaModel::load_from_file`).
  - Implements `TextToTextModel` for basic generation (`generate`, `generate_stream`).
  - Supports NUMA configuration via `LlamaCppSpecificConfig`.
  - Includes basic greedy sampling logic.
  - Uses `OnceCell` for global backend initialization.
  - Handles context creation per-request and uses a background thread for streaming to manage `LlamaContext` lifetimes.
  - Implemented `list_available_models` to scan `MODELS_PATH` for `.gguf` files following `provider/user/repo/filename.gguf` structure.
- Detailed `tracing` logs within `LlamaCppModel::generate_stream` for easier debugging.
- `ROADMAP.md` to track future work.

### Fixed
- **`diffusion-rs`**: Resolved `UninitializedField("prompt")` error during preset model loading by providing a dummy prompt to `diffusion_rs::preset::PresetBuilder`.
- **`diffusion-rs`**: Corrected test discovery for `diffusion_rs_integration.rs` by specifying the package with `cargo test -p inference-lib-examples ...`.
- **`diffusion-rs`**: Addressed `non_exhaustive` warnings for `diffusion_rs::preset::Preset` pattern matching in internal helper functions by adding wildcard `_` arms.
- Resolved various API compatibility issues between the initial implementation and `llama-cpp-2` v0.1.92 API, including:
  - `NumaStrategy` naming.
  - Parameter struct creation for `LlamaContextParams` (using setters like `with_n_ctx`). (Note: Similar fixes for `LlamaModelParams` regarding `mmap`/`mlock` failed and were commented out - see Known Issues).
  - Tokenization method (`str_to_token` instead of `tokenize`).
  - Sampling/candidate access (`logit`, `id`).
  - EOS token checking (`token_eos`).
  - Token-to-string conversion (`token_to_str`).
- Corrected model ID generation in `list_available_models` to match `provider:user/repo:filename` format.
- Fixed test failures caused by re-initializing `tracing_subscriber` multiple times by using `try_init()`.
- Resolved type errors in `llama_cpp` integration test setup.

### Changed
- **`diffusion-rs`**: Refactored preset handling in the `diffusion-rs` backend to dynamically list and parse all upstream `diffusion_rs::preset::Preset` variants. This replaced a manual, limited internal mapping and provides more comprehensive preset coverage.
- Updated root `README.md` and `ROADMAP.md` to reflect the new `diffusion-rs` backend, Text-to-Image capabilities, and related TODOs.
- Updated root `README.md` to mention `llama_cpp_2_rs` dependency for the planned `llama-cpp` backend.
- Refactored `LlamaCppService::load_model` to accept a model ID and resolve the file path internally using the `MODELS_PATH` environment variable, consistent with other backends.
- Updated `llama_cpp` integration tests to use the model ID convention.
- Updated `README.md` test instructions for `llama_cpp` backend.
- Added call to `LlamaBackend::void_logs()` during initialization to suppress verbose C++ library logging.
- Cleaned up various unused imports and functions across crates, including:
  - `std::path::PathBuf` from `diffusion_rs_integration.rs`.
  - `BackendConfig` from `crates/core/src/traits.rs`.
  - `str::FromStr` from `crates/backends/diffusion-rs/src/lib.rs`.
  - Commented out unused `map_from_diffusion_rs_sampler` in `diffusion-rs` backend.

### Known Issues
- **`diffusion-rs`**: `DiffusionRsModel` in `inference-lib-diffusion-rs` uses `unsafe impl Send + Sync` due to raw pointers in `diffusion-rs` internals. This requires auditing for thread safety. (Noted in `ROADMAP.md` and backend `README.md`).
- Configuration for `mmap` and `mlock` via `LlamaModelParams` setters (`with_mmap`, `with_mlock`) is currently commented out due to persistent, unexplained compiler errors despite matching `llama-cpp-2` v0.1.92 documentation. See `ROADMAP.md`.
- Sampling is currently limited to basic greedy search.
- Running `llama_cpp` integration tests concurrently (default `cargo test` behavior) can cause hangs, likely due to internal state/concurrency issues in `llama.cpp`. Tests pass when run sequentially (`--test-threads=1`).

## [0.1.0] - YYYY-MM-DD

### Added
- Initial project structure.
- Core traits (`InferenceService`, `Model`, `TextToTextModel`).
- Configuration structs (`BackendConfig`, `ApiConfig`, `LocalBackendConfig`, `GenerationOptions`).
- Unified `InferenceError` type.
- OpenAI backend implementation (`openai` feature).
- Anthropic backend implementation (`anthropic` feature).
- Integration tests (`inference-lib-examples` package).
- Basic README and project documentation.

[Unreleased]: https://github.com/your-username/your-repo-name/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-username/your-repo-name/releases/tag/v0.1.0 