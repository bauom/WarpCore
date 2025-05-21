# Inference Library Overview

This library (`warpcore`) provides a unified, type-safe interface in Rust for interacting with various inference backends, including both cloud-based APIs (like OpenAI, Anthropic) and local LLM engines (like Llama.cpp).

## Goal

The primary goal is to abstract away the differences between various AI inference providers and engines. This allows developers to write code once and easily switch between different backends or models (e.g., move from OpenAI's API to a locally hosted Llama model) with minimal code changes, often just by changing a backend type identifier and a model identifier.

## Core Concepts

- **Unified Interface:** A set of common traits (`InferenceService`, `Model`, `TextToTextModel`, etc.) define the core operations like loading models and generating text.
- **Backend Crates:** Each supported backend (OpenAI, Anthropic, Llama.cpp) is implemented in its own crate within the `crates/backends/` directory. This keeps the core library lean and allows users to only compile the dependencies they need.
- **Feature Flags:** Cargo features (`openai`, `anthropic`, `llama_cpp`) control which backend crates are compiled and included in the main `warpcore` crate.
- **Configuration:** Typed structs (`BackendConfig`, `ApiConfig`, `GenerationOptions`) allow for configuring backend connections (API keys, base URLs) and generation parameters (max tokens, etc.). Configuration can often be loaded automatically from environment variables.
- **Helper Functions:** Top-level functions like `create_inference_service` and `generate_text` in the main `warpcore` crate provide convenient ways to quickly get started.

## Workspace Structure

The project is organized as a Cargo workspace:

- **`src/lib.rs`**: The main library crate that re-exports functionalities from core and enabled backend crates. Provides top-level helper functions.
- **`crates/core/`**: Defines the core traits, configuration structs, error types, and common logic.
- **`crates/backends/`**: Contains individual crates for each supported inference backend.
  - `openai/`
  - `anthropic/`
  - `llama_cpp/` (In progress)
- **`crates/examples/`**: Contains integration tests (requires API keys/model paths to run).
- **`Cargo.toml`**: Defines the workspace, dependencies, and feature flags.

## Next Steps

- See `core.md` for details on the core abstractions.
- See `backends.md` for information on specific backend integrations and configuration.
- See `usage.md` for examples of how to use the library. 