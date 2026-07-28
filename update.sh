#!/bin/sh
# Pull the rolling image tags the config tracks (TRACK-LATEST policy -- see config.yaml).
# Re-run to advance every model to the newest llama.cpp / vLLM build. Per-entry KNOWN-GOOD
# fallback builds are recorded in config.yaml; if a fresh pull regresses a model, pin that
# entry to its known-good build until upstream fixes it.
docker pull ghcr.io/mostlygeek/llama-swap:cuda        # llama.cpp entries: Qwen, gemma, 122B, clean-2b, bge
docker pull ghcr.io/ggml-org/llama.cpp:server-cuda13  # laguna-s-2.1-mainline (CUDA-13 rolling; has the laguna arch)
docker pull vllm/vllm-openai:latest                   # vLLM entries: nvfp4, embedding (version-sensitive -- see comments)
