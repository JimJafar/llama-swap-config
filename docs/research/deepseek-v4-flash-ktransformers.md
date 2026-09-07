# DeepSeek V4 Flash on KTransformers: feasibility inputs

Research date: 2026-09-01. This note deliberately separates source-backed requirements from estimates so it can be combined with a local machine profile.

## Bottom line

KTransformers has a real, current DeepSeek V4 Flash path, but it is still a narrow and comparatively young configuration. The supported route is the official `deepseek-ai/DeepSeek-V4-Flash-0731` mixed MXFP4/FP8 checkpoint, served by the KTransformers SGLang fork with CPU/GPU heterogeneous inference. KTransformers' own support matrix labels the model entry **“Needs smoke”**, even though the model-specific tutorial now says RTX 5090, RTX 4090, and RTX 3090 paths are validated. Treat a successful local smoke test as mandatory, not as a formality. ([support matrix](https://ktransformers.net/en/docs/support-matrix), [model tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md))

The documented single-GPU baseline is:

| Resource | Documented requirement | Practical reading |
|---|---:|---|
| GPU | 1× RTX 5090, 32 GB | Best-supported reference configuration. RTX 4090 and RTX 3090 architectures are also marked validated, so 24 GB is an evidenced lower VRAM point, but not the tutorial baseline. |
| CPU | x86-64, AVX2 + FMA minimum | AVX-512 or AMX improves throughput. Generic AMD-CPU validation is still absent from the public hardware matrix. |
| RAM | ≥200 GB | This is the hard capacity gate. A reported current implementation behaviour keeps CPU copies of all experts even when some are assigned to GPU, so extra VRAM cannot safely be assumed to reduce host-RAM need. |
| Storage | ~340 GB | Use this as the free-space requirement. The Hugging Face repository itself reports 166.9 GB of stored data, so the KTransformers figure includes substantial operational/download headroom. |
| Software | Linux x86-64; CUDA 12.8+; FlashInfer ≥0.6.9; `transformers==4.57.1`; TileLang on non-Hopper GPUs | This is a pinned stack, not a normal “latest packages” install. A purpose-built Docker image is documented. |

Those requirements come from the current [KTransformers V4 Flash tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md). The general [CPU/GPU requirements page](https://ktransformers.net/en/docs/hardware-platforms/cpu-gpu-requirements) says Linux x86-64, NVIDIA Ampere or newer, AVX2 minimum, and that AMD CPU support remains a supported direction without a public model-validation combination.

## Exact model identity and checkpoint format

The current checkpoint to test is [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), not an unofficial GGUF and not the older preview checkpoint. DeepSeek describes 0731 as the official release that supersedes the preview while retaining the same model structure; KTransformers' current tutorial downloads this exact repository. ([DeepSeek model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/README.md), [KTransformers tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md))

DeepSeek describes the V4 Flash architecture as **284B total parameters, 13B activated per token, and a 1M-token maximum context**. Hugging Face's metadata for the 0731 repository displays **304B parameters**; 0731 includes the attached speculative-decoding module, and the public pages do not reconcile the two counting conventions. That discrepancy does not affect the capacity verdict, which uses the checkpoint's measured storage and KTransformers' own RAM requirement. The instruct checkpoint uses a mixed **FP4 + FP8** format: MoE expert weights are FP4 and the other parameters are FP8. ([official V4 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/README.md), [0731 repository](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), [DeepSeek API update confirming 0731 retains the preview architecture](https://api-docs.deepseek.com/updates/))

The official [0731 `config.json`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json) gives the relevant shape:

- `DeepseekV4ForCausalLM`, 43 layers, hidden size 4,096.
- 256 routed experts plus one shared expert; six routed experts selected per token.
- MoE intermediate size 2,048.
- One-million-token maximum (`max_position_embeddings=1048576`).
- `expert_dtype: fp4`; the remaining quantization config is FP8 E4M3 with UE8M0 scales.
- One attached next-token prediction layer for speculative decoding.

KTransformers v0.6.2 introduced native loading of the checkpoint's E2M1/UE8M0 expert representation without offline conversion. Its current support matrix calls this method `MXFP4` and marks it “Current, narrow” specifically for V4 Flash. ([v0.6.2 release](https://github.com/kvcache-ai/ktransformers/releases/tag/v0.6.2), [support matrix](https://ktransformers.net/en/docs/support-matrix))

## Capacity and bandwidth implications

### Host RAM

KTransformers explicitly requires at least 200 GB RAM for its validated single-5090 recipe. This is consistent with the official Hugging Face API reporting `usedStorage = 166,888,735,421` bytes for the 0731 repository, or **166.9 GB / 155.4 GiB**, before runtime buffers and duplicated representations. ([Hugging Face model API](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash-0731), [KTransformers tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md))

There is also a still-open issue against KTransformers reporting that the CPU backend allocates all 256 experts per layer even when many experts are GPU-resident. In that report, a 96 GB RTX PRO 6000 plus 123 GB RAM still used about 170 GB of anonymous memory and thrashed swap; changing `--kt-num-gpu-experts` did not lower host allocation. This is user-reported evidence rather than a maintainer guarantee, but it matches the official ≥200 GB requirement and means **do not count VRAM plus RAM as one fungible pool** on the stock path. ([issue #2084](https://github.com/kvcache-ai/ktransformers/issues/2084))

### Disk

The checkpoint repository's current stored size is about 167 GB, but KTransformers specifies about 340 GB storage for the model workflow. The conservative go/no-go test should therefore be **340 GB genuinely free on fast local storage**, preferably NVMe; the 1M-context recipe explicitly calls for NVMe for weight-loading speed. ([Hugging Face model API](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash-0731), [single-GPU tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md), [1M-context recipe](https://ktransformers.net/en/docs/inference/long-context-deployment))

### CPU memory bandwidth

The CPU is not merely a weight store; it evaluates the offloaded routed experts on every token. A lower-bound calculation from the official config illustrates the scale:

```text
43 layers × 6 active experts × 3 matrices × 4096 × 2048
= 6.493 billion active routed-expert weights/token

At 4 bits/weight: 3.247 GB of routed-expert weights/token
With 10 of 256 experts resident on GPU and uniform routing:
3.247 × (246/256) ≈ 3.12 GB/token read by the CPU side
```

This ignores scales, activations, cache effects, routing skew, and non-expert work, so it is an optimistic traffic estimate. Sustaining 10 tokens/s would imply roughly **31 GB/s** of useful host expert-weight bandwidth; 20 tokens/s roughly **62 GB/s**. Consequently, merely having 200+ GB RAM is insufficient for a pleasant result if the memory subsystem is slow or poorly populated. The official 1×5090 reference reports **20+ tok/s**, but that performance number should not be transferred to a different CPU, memory-channel count, or NUMA topology. ([config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json), [KTransformers single-GPU result](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md))

## GPU and context practicality

KTransformers keeps routed experts on CPU and GPU while the attention path and KV cache consume GPU memory. Its current tutorial's default is only **16,384 tokens**, `--max-running-requests 2`, and ten GPU experts on a 32 GB RTX 5090. That is the practical everyday starting point, despite the model advertising 1M context. ([launch recipe](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md))

The official 1M-token deployment is a separate high-end recipe: **4× RTX 5090 (128 GB aggregate VRAM), dual-socket 64-core AVX-512 CPU, 256 GB DDR5, PCIe 5.0, and NVMe**. It uses tensor parallelism 4 and reports roughly 5.34 GB of GPU headroom after its KV pool and CUDA graph allocation. A single consumer GPU should not be assessed on the assumption that the full 1M context is practical. ([1M-context recipe](https://ktransformers.net/en/docs/inference/long-context-deployment))

For normal local use, assess feasibility at 16K context first. A 24 GB RTX 3090 is now marked validated and KTransformers v0.6.4 added Ampere fallbacks, but a 32 GB RTX 5090 remains the documented performance baseline. NVIDIA Ampere/Ada/Blackwell are the active path; ROCm is classified as legacy rather than current support. ([model tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md), [v0.6.4 release](https://github.com/kvcache-ai/ktransformers/releases/tag/v0.6.4), [hardware requirements](https://ktransformers.net/en/docs/hardware-platforms/cpu-gpu-requirements))

## Software and maturity risks

- The current model tutorial requires CUDA 12.8+, matching FlashInfer Python and cubin packages at 0.6.9 or later, and exactly `transformers==4.57.1`; Transformers 5.x is documented to break the V4 config import.
- Non-Hopper GPUs need TileLang; the tutorial pins `tilelang==0.1.8` as validated and requires `apache-tvm-ffi<0.1.12` to avoid a runtime collision.
- The official Docker quick start (`approachingai/ktransformers:DSV4-specific`) avoids much of the build complexity, but it is a model-specific image rather than a generic stable deployment.
- KTransformers' support matrix still says “Needs smoke”. There have been architecture-specific failures in the issue tracker, reinforcing the need to validate the exact GPU/CPU/package combination rather than assuming that a broad architecture label guarantees success. ([tutorial prerequisites](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md), [support matrix](https://ktransformers.net/en/docs/support-matrix), [example Hopper issue #2077](https://github.com/kvcache-ai/ktransformers/issues/2077))

## Go/no-go rubric for the local profile

| Local machine finding | Assessment |
|---|---|
| NVIDIA Ampere/Ada/Blackwell, ≥24 GB VRAM; ≥200 GB usable RAM; ≥340 GB free NVMe; x86 AVX2+FMA | Worth a 16K-context smoke test. 32 GB Blackwell plus fast, many-channel RAM is the strongest single-GPU match. |
| 192 GB installed RAM | Below the official minimum and likely tight after the OS/runtime; not a practical stock configuration even if it might sometimes start. |
| 128 GB RAM or less | No-go on the current stock path; swap would make interactive decoding impractical. |
| AMD GPU / ROCm, Intel GPU-only, Apple Silicon, or non-x86 host | No current documented V4 Flash path. |
| NVIDIA GPU below Ampere or clearly below 24 GB VRAM | Outside the evidenced V4 Flash configurations; do not call it practical without an independent successful recipe. |
| Slow or low-channel host memory | Capacity may be sufficient but decode can still be poor because routed-expert inference is bandwidth-heavy. |
| Expectation of 1M local context on one GPU | No-go; the official 1M recipe is four RTX 5090s plus a dual-socket 256 GB host. |

## Recommended first experiment if the machine clears the gates

Use the model-specific Docker image and the exact 0731 checkpoint at its default 16K context, two concurrent requests, and conservative GPU-memory fraction. This is the shortest path to a meaningful smoke test and preserves the current package pins. Only after it generates correct output should the native build, larger context, more GPU experts, dynamic expert placement, or speculative decoding be tuned. The tutorial reports 4–5 minutes for startup on the 5090 reference, so startup time alone is not evidence of a hang. ([Docker and launch instructions](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md))
