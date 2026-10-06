#!/bin/sh
# Report the installed llama-swap (host binary) and llama.cpp (official ggml-org image)
# versions. Does not depend on any model being loaded.
IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda13
echo "llama.cpp:  $(docker run --rm --entrypoint /app/llama-server "$IMAGE" --version 2>&1 | grep -i version | head -1)"
echo "llama-swap: $(llama-swap --version 2>&1 | head -1)"
