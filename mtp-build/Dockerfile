# Builder: compile danielhanchen/llama.cpp@qwen4exp/mtp (Flash-Next MTP) against CUDA 13
# so the resulting llama-server links a cudart.so.13 compatible with the runtime image.
FROM nvidia/cuda:13.1.0-devel-ubuntu24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    pciutils build-essential cmake curl git libcurl4-openssl-dev ca-certificates && \
    rm -rf /var/lib/apt/lists/*
ARG BRANCH=qwen4exp/mtp
RUN git clone --branch "${BRANCH}" --depth 1 \
    https://github.com/danielhanchen/llama.cpp && \
    cd llama.cpp && echo "BUILD_COMMIT=$(git rev-parse HEAD)" > /llama.cpp/build-info.txt
WORKDIR /llama.cpp
RUN cmake . -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
    -DGGML_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j --clean-first \
      --target llama-server llama-mtmd-cli llama-cli llama-gguf-split && \
    cp build/bin/llama-server /llama-server && \
    cp build/bin/llama-cli /llama-cli && \
    cp build/bin/llama-mtmd-cli /llama-mtmd-cli && \
    cp build/bin/llama-gguf-split /llama-gguf-split

# Runtime: reuse the existing CUDA-13 server image's CUDA runtime + /app layout,
# swap in our Flash-Next MTP-capable llama-server.
FROM ghcr.io/ggml-org/llama.cpp:server-cuda13-b10775
COPY --from=build /llama-server /app/llama-server
COPY --from=build /llama.cpp/build-info.txt /app/build-info.txt
ENTRYPOINT ["/app/llama-server"]
