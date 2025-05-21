# Roadmap / TODO

This document tracks upcoming milestones, tasks, and open questions for the **Inference Library**.

> Legend: ✅ Done · 🚧 In Progress · 💤 Blocked · 📝 Design/Planned · ❓ Needs Discussion

---

## v0.1.0 (Current Focus)

- [✅] Core crate with traits, configs, error types
- [✅] OpenAI backend (chat completion API)
- [✅] Anthropic backend (messages API)
- [✅] Streaming generation API (`Stream<Item = Result<String>>`)
- [✅] Integration tests for OpenAI & Anthropic
- [✅] `diffusion-rs` backend (local text-to-image generation)
- [✅] `TextToImageModel` trait and core support for image generation.
- [🚧] `llama.cpp` backend (CPU + GPU layers) - Basic implementation complete, needs refinement.
- [📝] CI setup (lints, tests, potentially ignored tests with secrets)
- [📝] Crate-level docs (`cargo doc --all-features`)
- [📝] Basic `tracing` instrumentation

---

## v0.2.0

- [📝] Configurable on-disk prompt cache (e.g., using Moka)
- [📝] Add usage statistics (token counts) to responses/events
- [📝] Helpers for auto-discovering local model files
- [📝] Improved error handling and reporting details
- [📝] Release initial version on crates.io

---

## v0.3.0

- [📝] `candle-rs` backend
- [📝] `groq` backend
- [📝] `modal` backend
- [📝] `ImageToTextModel` trait + OpenAI Vision implementation

---

## v0.4.0

- [✅] `TextToImageModel` trait + placeholder/basic implementation
- [📝] Prompt template library/integration
- [📝] Retry & back-off middleware for cloud backends
- [📝] More comprehensive `tracing`/OpenTelemetry support

---

## v1.0.0

- [📝] Stabilise public API & mark crates `1.0`
- [📝] Complete feature flags coverage & documentation
- [📝] Expanded regression test-suite (potentially with fixture models/mock servers)
- [📝] Benchmarks

---

## Nice-to-haves / Ideas

- [❓] WASM support for inference in browsers (WebGPU)
- [❓] Python bindings via PyO3
- [❓] `xtask` command for releasing & validation
- [❓] Plugin system for community backends

---

## Future Backends

- [ ] `candle-rs`
- [ ] `groq` / `modal`

## Core Improvements

- [ ] More detailed tracing/observability
- [✅] Support for `ImageToText` and `TextToImage` model types
- [ ] Enhanced configuration validation

## Backend Specific

### `llama_cpp` Backend
- [✅] Implement `list_available_models` by scanning `MODELS_PATH`.
- [🚧] Implement advanced sampling strategies (beyond greedy).
- [🚧] Add integration tests.
- [🚧] Improve error handling.
- [❓] Investigate and document concurrency limitations (tests hang unless run with `--test-threads=1`).

### `diffusion-rs` Backend
- [❓] **Audit Thread Safety**: The `DiffusionRsModel` currently uses `unsafe impl Send + Sync` due to raw pointers in `diffusion-rs` internals. This needs careful review and an attempt to remove `unsafe` if possible, or confirm and document the safety implications. (See `crates/backends/diffusion-rs/README.md`)
- [📝] **Sampler Mapping**: The `map_from_diffusion_rs_sampler` is currently unused (commented out). Decide if it's needed for future features or remove it.
- [📝] **Progress Callbacks**: Expose `diffusion_rs` progress callbacks for a better user experience during long generation tasks, potentially via a streaming API for progress updates.
- [📝] **Output Formats**: While `ImageOutputFormat` exists, the backend currently only directly produces PNG files. Add support for returning raw bytes and other formats like JPEG/WEBP if `diffusion-rs` supports them or if we add conversion.
- [📝] **List Local Models**: Enhance `list_available_models` to scan `$DIFFUSION_MODELS_PATH` for local model files, similar to the `llama_cpp` backend.
- [📝] **BackendConfig**: Respect user-supplied `BackendConfig` for potential future options like device selection or thread counts if `diffusion-rs` exposes such controls.

## Known Issues

- See backend-specific items above.

Feel free to open issues or PRs to suggest additions / claim tasks! 