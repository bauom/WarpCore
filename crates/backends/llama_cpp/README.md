# Llama.cpp Backend

This backend integrates [llama_cpp_2](https://github.com/flaneur2020/llama-cpp-rs) (version 0.1.92) with the inference library, providing a local inference option for LLM models compatible with llama.cpp.

## Current Status

The backend is currently in development with the following progress:

- [x] Basic structure for implementing the InferenceService and TextToTextModel traits
- [x] Loading models with configuration options
- [x] NUMA strategy support
- [ ] Complete compilation with the current API version

## Known Issues

The code is experiencing API compatibility issues with `llama_cpp_2` v0.1.92. Key issues include:

1. The API structure is different from what was expected:
   - `LlamaContextParams` and `LlamaModelParams` use builder pattern instead of direct field assignments
   - Some methods like `kv_cache_seq_rm` don't exist or have different signatures
   - Naming and token handling methods are different (using `tokenize`, `token_to_piece` instead of `str_to_token`, etc.)
   - The sampler API has changed significantly

2. The current implementation attempts to provide basic functionality with:
   - Simple greedy sampling (picking highest probability token) since the sampling API appears to be different
   - Working around missing `kv_cache_seq_rm` method
   - Implementing stream generation support with a background thread

## Next Steps

1. Explore the [llama_cpp_2 documentation](https://docs.rs/llama_cpp_2/0.1.92/llama_cpp_2/) to identify the correct API structure
2. Consider updating to a newer version if available and compatible
3. Check examples in the library repository to understand proper usage patterns
4. Implement proper sampling functionality using the available API
5. Improve error handling and recovery for more robust operation

## Configuration Options

The backend supports these configuration options:

- `context_size`: Maximum context size for the model
- `threads`: Number of threads to use for inference
- `gpu_layers`: Number of layers to offload to GPU
- `use_mmap`: Whether to use memory-mapped loading
- `use_mlock`: Whether to lock memory
- `numa_strategy`: NUMA strategy ("DISABLED", "NUMACTL", "DISTRIBUTE", "MIRROR")

See `LlamaCppSpecificConfig` in core crate for additional options. 