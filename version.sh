#!/bin/sh
# Print the llama.cpp and llama-swap versions inside the running container.
IMAGE=ghcr.io/mostlygeek/llama-swap:cuda
C=$(docker ps -q --filter ancestor="$IMAGE" | head -1)
if [ -z "$C" ]; then
  echo "No running container for $IMAGE"
  exit 1
fi
echo "llama.cpp:  $(docker exec "$C" llama-server --version 2>&1 | head -1)"
echo "llama-swap: $(docker exec "$C" /app/llama-swap --version 2>&1 | head -1)"
