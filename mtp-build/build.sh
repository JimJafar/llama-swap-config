#!/usr/bin/env bash
# Build llama.cpp CUDA server with the qwen4exp MTP draft-head patch.
# Base: unslothai/llama.cpp@bea3b12d (verified by dzannotti's Qwen3.8-Flash-Next-MTP-GGUF)
# Apply: qwen4exp-mtp-draft-head.patch (PR #27739 MTP graph)
# Output: local/qwen38-flash-mtp:server-cuda13
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$ROOT/build-mtp"
mkdir -p "$WORK"
cd "$WORK"
if [ ! -d llama.cpp-b10612/.git ]; then
  git clone --depth 1 --branch b10612 https://github.com/ggml-org/llama.cpp.git llama.cpp-b10612
fi
cd llama.cpp-b10612
# Switch to the pinned base that carries PR #27742 (qwen4exp model) and apply the MTP patch
git fetch --depth 1 origin bea3b12daee45876b0129a3602dc8f534ce30bf0
git checkout -q bea3b12daee45876b0129a3602dc8f534ce30bf0
git apply "$ROOT/mtp-build/qwen4exp-mtp-draft-head.patch"
docker build -f .devops/cuda.Dockerfile --target server \
  --build-arg CUDA_VERSION=13.3.0 --build-arg UBUNTU_VERSION=24.04 --build-arg GCC_VERSION=14 \
  -t local/qwen38-flash-mtp:server-cuda13 .
