#!/bin/sh
# Report the installed llama-swap (host binary) and llama.cpp (bundled image)
# versions. Does not depend on any model being loaded.
IMAGE=ghcr.io/mostlygeek/llama-swap:cuda
echo "llama.cpp:  $(docker run --rm --entrypoint /app/llama-server "$IMAGE" --version 2>&1 | grep -i version | head -1)"
echo "llama-swap: $(llama-swap --version 2>&1 | head -1)"
