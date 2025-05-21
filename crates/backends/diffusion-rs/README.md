# `inference-lib` – Diffusion-rs Backend

This crate (`inference-lib-diffusion-rs`) wires the [`diffusion-rs`](https://crates.io/crates/diffusion-rs) project into the **`inference-lib`** umbrella so that you can generate images from text with the same high-level API that powers the other back-ends (OpenAI, Llama-CPP, …).

---

## 1  How it works

*   **New core abstractions** – we added
    * `ModelType::TextToImage` and `BackendType::DiffusionRs` in `crates/core`.
    * `DiffusionOptions`, `ImageOutput`, `ImageOutputFormat`, and the `TextToImageModel` trait so back-ends can expose rich generation controls.
*   **Backend crate** –  `crates/backends/diffusion-rs` implements
    * `DiffusionRsService` that fulfils the `InferenceService` trait.
    * `DiffusionRsModel` that fulfils both `Model` and `TextToImageModel`.
    * `load_model` understands either
        * a **preset** name (e.g. `"SDXLTurbo1_0Fp16"`) that is mapped to a `diffusion_rs::preset::Preset`, **or**
        * a **local file-path** (absolute or resolved against `$DIFFUSION_MODELS_PATH` → `$MODELS_PATH`).
    * Generation delegates to `diffusion_rs::api::txt2img`.
*   **Cargo feature flag** – compile the back-end with `--features diffusion-rs`.

```console
# Workspace-wide build
cargo test --all-features                   # enable everything
# or a selective build
cargo test -p inference-lib-examples --features diffusion-rs
```

---

## 2  Using the back-end in code

```rust
use inference_lib::{
    BackendType, create_inference_service, ModelType,
    DiffusionOptions, SamplerKind, ImageOutput
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. choose the backend
    let svc = create_inference_service(BackendType::DiffusionRs, None).await?;

    // 2. load a model – preset or local path
    let model = svc.load_model("SDXLTurbo1_0Fp16", ModelType::TextToImage, None).await?;
    let txt2img = model.as_text_to_image().expect("not a diffusion model");

    // 3. ask for an image
    let opts = DiffusionOptions::new()
        .with_width(512)
        .with_height(512)
        .with_steps(8)
        .with_sampler(SamplerKind::EulerA)
        .with_seed(42);

    match txt2img.generate_image("A corgi in a wizard hat", Some(opts)).await? {
        ImageOutput::File(p) => println!("Image saved at {}", p.display()),
        ImageOutput::Bytes(_, _) => println!("Received in-memory bytes"),
    }
    Ok(())
}
```

### Environment variables

* `DIFFUSION_MODELS_PATH` – a folder that contains your local weights (`.safetensors`, `.ckpt`, **or the quantised `.gguf` format produced by stable-diffusion.cpp**).
* `MODELS_PATH` – fallback if the var above is not set.

---

## 3  Tests & examples

A full integration test lives in
`crates/examples/tests/diffusion_rs_integration.rs` and is run automatically when the `diffusion-rs` feature is enabled.

Two scenarios are asserted:

1. **Preset generation** (always on, downloads weights on-the-fly).
2. **Local model generation** (ignored by default – update the hard-coded path or set the env-vars to enable).

---

## 4  Safety & caveats

* `DiffusionRsModel` contains raw pointers inside the `diffusion_rs` FFI structs. We applied `unsafe impl Send + Sync` **temporarily** so the type can cross async thread boundaries. **This has _not_ been audited** – use with care.
* The back-end only guarantees PNG output for now; other formats are forwarded but not converted.
* Only the sampler variants that exist in `diffusion-rs v0.1.9` are mapped. Asking for an unknown sampler will currently panic.

---

## 5  Roadmap / TODOs

- [ ] Audit thread-safety of `DiffusionRsModel` and remove the `unsafe impl` if possible.
- [ ] Implement reverse sampler mapping (remove the commented-out helper or make it used).
- [ ] Expose progress callbacks (`sd_set_progress_callback`) via streaming API.
- [ ] Add support for additional image formats (JPEG/WEBP) and byte-buffer output directly.
- [ ] Auto-scan `$DIFFUSION_MODELS_PATH` for available local models and return them from `list_available_models`.
- [ ] Respect user-supplied `BackendConfig` (device selection, thread-count, etc.).
- [ ] Add upscaler / img2img / ControlNet hooks once `inference-lib` grows corresponding traits.
- [ ] Write end-to-end example binary in `crates/examples/src/bin/diffusion_cli.rs`. 