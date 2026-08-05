# Local LLM inference optimisation — structured guide extraction

> Source: [Local LLM Inference Optimization: The Complete Guide](https://carteakey.dev/blog/local-inference/local-llm-optimization/) ([author-provided raw text](https://carteakey.dev/blog/local-inference/local-llm-optimization/raw.txt)) by Kartikey Chauhan, published 12/06/2026 and updated through 17/07/2026.
>
> Extracted and checked: 02/08/2026.
>
> This is a condensed, paraphrased Markdown reference rather than a verbatim copy. Keep the source link with it because the guide is actively maintained.

## Scope and evidence

The guide is about direct llama.cpp tuning on one consumer CUDA workstation: an RTX 4070 12 GB, an Intel i5-12600K, DDR5-6000, Linux, and recent llama.cpp builds. Its measured results are useful leads, not universal constants—especially for AMD, Apple Silicon, multi-GPU systems, and newer model architectures.

The author distinguishes three levels of evidence:

- **Tested here:** measured on the author's RTX 4070 node.
- **Upstream behaviour:** checked against current llama.cpp documentation.
- **Needs testing:** a plausible lead that has not yet earned a recommendation.

## Start with the symptom

| Symptom | Measure first | Change next |
| --- | --- | --- |
| Fails to load or OOMs later | Free VRAM and VRAM after realistic context growth | Reduce context or parallel slots, quantise KV, increase fit headroom, then revisit placement |
| Slow prompt ingestion | Prompt-processing throughput on a representative long prompt | Sweep micro-batch size and verify GPU offload/backend |
| Slow token generation | TG, RAM speed, CPU frequency and placement | Correct memory/power state, then test placement and P-core affinity |
| Speculation adds no speed | Acceptance, TG and complete request latency | Shorten the draft and compare target/draft KV precision |
| Two GPUs are slower than one | Split mode, NCCL and interconnect | Start with layer split; treat tensor split as experimental |

For coding agents, a decode-only benchmark is insufficient. Record time to first token (TTFT), prompt processing (PP), token generation (TG), prompt-cache behaviour, and at least one full tool loop.

## Safe starting profiles from the guide

These are starting points measured on the author's system, not capacity rules for Marvin.

| Workload | Fit headroom | Context | KV cache | Slots | Batch |
| --- | ---: | ---: | --- | ---: | ---: |
| Text on 12 GB | 512 MiB | 64K | q8 K/V | 1 | 1024 |
| Text on 24 GB | 512–768 MiB | 128K | q8 K/V | 1–2 | 1024 |
| Vision on 12 GB | at least 512 MiB | 64K | q8 K/V | 1 | 256 |
| MTP | at least 512 MiB | 64K | benchmark per model | 1 | 1024 |

## Changes that moved the needle on the source machine

| Change | Observed result or rationale |
| --- | --- |
| Enable the RAM's rated XMP profile | Restored severely degraded MoE generation speed |
| Keep Intel E-cores out of the inference thread set | Improved affected profiles by roughly 20–30% |
| Use q8 KV and one slot | Freed VRAM for more GPU-resident weights |
| Gemma 4 QAT plus MTP | Produced a model-specific 2.0–2.6× TG improvement |
| Let `--fit` choose initial placement | Current layer-split llama.cpp can size placement from real free VRAM |
| Test n-gram speculation, CUDA graph optimisation and tensor parallelism | Keep them only when a complete workload beats baseline |

## Measurement contract

Capture the following for every candidate profile:

| Metric | What it reveals |
| --- | --- |
| Cold and warm TTFT | Model load, prompt processing and cache effects |
| PP tokens/s | Batch, GPU kernels, placement and long-prompt cost |
| TG tokens/s | Weight bandwidth, RAM bandwidth and CPU affinity |
| VRAM after load | Weights, KV, compute buffers and projector fit |
| VRAM after a long session | VMM/CUDA graph growth and inadequate headroom |
| RAM usage and swap deltas | Whether hybrid weights are paging or swapping |
| MTP acceptance | Whether draft work is actually useful |
| Prompt-cache hits | Whether repeated agent context avoids re-prefill |
| Full tool-loop time | The latency users actually experience |

Do not optimise from a short prompt. Include cold start, warm cache, a normal inspect/edit/explain loop, and a prompt close to the context length actually served.

Minimum provenance to save:

```text
model and quant:
llama.cpp build/commit or image digest:
complete command:
context and parallel slots:
batch / micro-batch:
target and draft KV types:
prompt-cache state:
PP / TG / TTFT:
draft acceptance:
VRAM at load / after long session:
RAM and swap before / after:
notes and failures:
```

## Hardware and memory hierarchy

Decode generally streams active weights through the memory hierarchy. GPU VRAM is much faster than system RAM; system RAM is much faster than storage. Dense models want all weights in VRAM. Hybrid MoE models may deliberately keep experts in RAM, so memory channels and actual configured RAM speed can dominate TG.

Checks to run before flag tuning:

```bash
sudo dmidecode -t memory | grep -E "Speed|Configured"
```

Confirm that configured speed matches the intended XMP/EXPO profile.

The guide suggests moving the desktop display from an NVIDIA GPU to integrated graphics to reclaim roughly 0.5–1 GB of NVIDIA VRAM. This only applies when the CPU actually provides an iGPU and the desktop/display stack works through it.

For fully GPU-resident dense models, CPU count matters little. For CPU-offloaded MoE, test P-core-only execution on hybrid Intel CPUs rather than assuming all cores are faster.

## Linux and host tuning

Apply one change at a time and preserve a baseline.

### Power state

Check the governor, energy-performance preference, and observed frequency:

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference
grep "cpu MHz" /proc/cpuinfo | sort -rn | head -6
```

The author's Intel machine sometimes benchmarked 20–30% low under `power-profiles-daemon`, even when obvious checks looked reasonable. `tuned-ppd` fixed that machine. Treat this as a diagnosis to A/B test, not a blanket reason to replace a working desktop power manager.

### Transparent huge pages

```bash
cat /sys/kernel/mm/transparent_hugepage/enabled
```

The source machine uses `always`, but the guide recommends testing rather than assuming.

### Headless mode

Stopping the graphical target can free compositor RAM and VRAM:

```bash
sudo systemctl isolate multi-user.target
# restore later
sudo systemctl isolate graphical.target
```

This ends the graphical session and is an operational mode switch, not a harmless inference flag.

## Runtime and reproducibility

Choose the runtime before flags:

- **llama.cpp:** maximum control over placement, KV, batching and backend behaviour.
- **Ollama:** simpler model management with a smaller tuning surface.
- **LM Studio:** desktop model browsing and a convenient local endpoint.
- **vLLM:** continuous batching and multi-user throughput when the model/runtime fit the hardware.
- **exllamav2:** dense CUDA models that fit fully in VRAM.
- **MLX:** Apple Silicon.

Record an exact llama.cpp commit or container digest for every benchmark. A moving package or image tag makes an otherwise careful comparison irreproducible.

Useful llama.cpp binaries include `llama-server`, `llama-bench`, `llama-fit-params`, `llama-cli`, and `llama-sweep-bench`.

## Model and quant selection

Dense and MoE models need different tuning strategies:

| | Dense | MoE / hybrid |
| --- | --- | --- |
| Active weights per token | Most or all | Selected experts plus shared components |
| Best performance | Fully in VRAM | Attention/shared tensors on GPU; experts may be in RAM |
| Main tuning lever | Quant and context | Placement, RAM bandwidth and CPU affinity |

Use the highest quant that fits the real serving budget and wins on task quality. The guide treats Q5/UD-Q5 as a strong default, Q4 as the usual size/quality compromise, Q6/Q8 as high-headroom options, and IQ/importance-aware quants as workload-dependent rather than automatically better.

For a meaningful quality choice, compare perplexity or KL divergence and actual target tasks. Quantisation-aware training can let a low-bit model recover quality, but the gain is model-specific.

## Placement

For dense models that fit, offload all layers. For hybrid MoE models, placement is usually the dominant tuning problem.

- `--n-gpu-layers` is coarse layer placement.
- `--n-cpu-moe` keeps a chosen count of MoE expert layers on CPU.
- `--override-tensor` provides regex-based, per-tensor placement.
- `--fit on` probes free VRAM and chooses unset placement parameters in supported split modes.
- `--fit-target` reserves device headroom; a tiny margin may load successfully and still OOM as context grows.
- `llama-fit-params` can turn an exploratory fit into reproducible static flags.

Shared-expert tensors need to be included when writing manual MoE tensor patterns. A pattern that catches routed experts but leaves always-active shared experts on GPU can silently consume the expected headroom.

## Context, KV cache and concurrency

KV usage grows with context and with the number of parallel slots. Test at the real served context, not a 512-token micro-benchmark.

- `-ctk q8_0 -ctv q8_0` is the guide's text-server starting point on CUDA.
- q4 KV is a pressure-relief option with greater quality risk.
- Each parallel slot needs its own KV allocation; single-user servers commonly benefit from `--parallel 1`.
- Flash Attention is important for large-context memory use and is mandatory for current tensor-split mode.
- Hybrid and linear-attention models can have very different KV scaling; measure rather than applying generic GB-per-context tables.

## Batch and micro-batch

- `--batch-size` is the logical prompt-processing batch.
- `--ubatch-size` is the physical micro-batch that drives compute-buffer size.
- Larger values can improve PP until memory pressure or kernel behaviour reverses the gain.
- Sweep realistic long prompts. A short prompt can conceal an OOM that occurs only when the full prefill graph is reserved.
- Keep the physical micro-batch no larger than the logical batch.

## Sampling

Sampling changes output quality and style, not core inference throughput. Use model-card defaults as the starting point, then tune temperature, top-k, top-p, min-p and repetition penalties on an evaluation set. Do not mix sampling changes into a performance A/B.

## CPU affinity and process controls

Relevant llama.cpp controls include:

- `--threads` for generation.
- `--threads-batch` for prompt processing.
- `--cpu-range` / `--cpu-mask` and strict variants for affinity.
- Docker `--cpuset-cpus` or `taskset` for an outer hard boundary.
- `--poll` for busy-wait aggressiveness; the guide found it flat and recommends leaving it alone.
- NUMA modes matter on multi-socket hosts, not a single NUMA node.
- `--prio` can reduce scheduler jitter; measure tail latency as well as mean TG.

## Memory mapping and locking

- `--no-mmap` forces an up-front load and can remove page-fault jitter, especially during MoE prompt processing.
- `--mlock` prevents model pages from being swapped.
- Both increase the consequences of memory pressure. Large RAM-offloaded models may need evictable page cache, making `mmap` deliberate and `mlock` unsafe.

## CUDA experiments

The guide recommends A/B testing `GGML_CUDA_GRAPH_OPT=0` and `1` on a long, variable-context workload. Graph capture can help dispatch overhead but may increase memory use or regress some models. It also reports no gain from forcing cuBLAS over llama.cpp's consumer-oriented quant kernels on the tested Q4/MXFP4 workloads.

## Speculative decoding and MTP

Speculation is useful only when accepted tokens and end-to-end latency improve.

- Sweep draft length around the upstream default rather than copying a value from another model.
- Target (`-ctk`/`-ctv`) and draft (`-ctkd`/`-ctvd`) KV caches are separate decisions.
- The author's Gemma tests needed full-precision KV to preserve good acceptance, while Qwen may behave differently.
- Record acceptance, TG, VRAM and full request time together.
- N-gram speculation may help repetitive code, but it remains a workload-specific experiment.

## Vision

The multimodal projector consumes its own VRAM and compute-buffer headroom. Supply it during the fit probe. Image tokenisation can also exceed a small micro-batch, so a vision profile needs a realistic image test rather than a text-only load check.

## Multi-GPU

Current upstream llama.cpp distinguishes:

| Mode | Use |
| --- | --- |
| `layer` | Compatible default; contiguous layers on devices; less communication |
| `row` | Deprecated older row split |
| `tensor` | Experimental tensor parallelism; more communication and architecture constraints |

Start with layer split when the goal is capacity. Tensor split may improve latency when NCCL and the interconnect are good, but it can lose to layer split or fail on unsupported architectures.

Important current upstream constraints:

- `--fit` is not supported in tensor mode.
- Tensor mode requires Flash Attention.
- Tensor mode currently requires non-quantised KV (`f16`, `bf16` or `f32`).
- NCCL is selected at build time and used automatically when available.
- P2P is opt-in and can be unstable on unsupported consumer topologies.
- Validate output quality as well as speed because tensor mode is experimental.

See the current [llama.cpp multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md) before updating an older working TP profile.

## Security

Treat a local inference server as an HTTP service. Bind to localhost unless LAN access is intentional; use authentication and a private network or VPN for remote access; assume prompts and tool calls can reach logs; and record model provenance and hashes where practical.

## Diagnostic checklist

Before benchmarking:

1. Confirm configured RAM speed/XMP.
2. Record governor, EPP and observed P-core frequency.
3. Start with known free VRAM.
4. Check temperature and throttling under sustained load.
5. Check background CPU consumers.
6. Record swap counters before and after the run.
7. Verify PCIe generation and width under load.
8. Confirm the active power profile.
9. Save the exact runtime build and command.
10. Run long enough to expose context growth and memory drift.

## Primary references used by the guide

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [llama.cpp server reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [llama.cpp multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)
- [llama.cpp speculative decoding guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- [llama.cpp quantisation guide](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [L3MS benchmark toolkit](https://github.com/carteakey/l3ms)
