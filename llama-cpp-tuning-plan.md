# Marvin llama.cpp tuning plan

> Prepared 02/08/2026 from the current `config.yaml`, `models.json`, repository notes, live host topology, the [source tuning guide](https://carteakey.dev/blog/local-inference/local-llm-optimization/), and current upstream llama.cpp documentation. No runtime configuration has been changed by this plan.

## Bottom line

Most of the guide's high-impact llama.cpp advice is already present: CUDA/Linux, one slot, q8 KV where appropriate, Flash Attention, MTP, model-specific micro-batches, automatic fit for compatible models, deliberate mmap choices, GPU UUID pinning, and measured layer-versus-tensor split variants.

Two experiments are complete. **P-core-only execution was rejected for Qwen3.5-122B** because it reduced generation throughput by 16.2%. **DeepSeek V4 DSpark at 131K is a promising opt-in result:** fully warm generation reached 17.02–17.53 t/s at 69.7% acceptance, but the new runtime still has fit/placement bugs and greedy output was not bit-identical to the no-draft control. The best remaining experiments are **a real power-profile A/B** and **`--no-mmap` on Qwen3.5-122B now that the host has 64 GB RAM**. Before any other container update, pin the working builds: the current Qwen TP profile relies on behaviour that now conflicts with upstream tensor-mode constraints.

The iGPU display trick cannot be used on this machine as built. Marvin has a **Core Ultra 7 265KF**, which has **no integrated graphics**, and the live PCI inventory contains only the three NVIDIA GPUs. The motherboard's HDMI/USB-C display outputs work only with a CPU that has integrated graphics.

## Current machine facts

| Item | Observed state | Consequence |
| --- | --- | --- |
| CPU | Core Ultra 7 265KF: 8 P-cores + 12 E-cores, 20 threads | CPUs 0–7 are P-cores; CPUs 8–19 are E-cores |
| iGPU | None; Intel specifies 0 graphics cores and 0 supported displays | Motherboard HDMI cannot take over the desktop |
| Board | MSI MPG Z890 Carbon WiFi | Board display outputs still require a graphics-enabled CPU |
| GPUs | 5060 Ti at 01:00, 5070 Ti at 02:00, 5060 Ti at 03:00 | 48 GB total NVIDIA VRAM; 5070 owns the connected HDMI output |
| Display | `card2-HDMI-A-2` on the 5070 Ti is the only connected DRM output | Desktop composition and scan-out consume 5070 VRAM |
| CPU policy snapshot | `intel_pstate`, governor `powersave`, EPP `balance_performance` | Worth an A/B; not proof that the CPU is currently slow |
| THP | `always` | Already matches the guide's source-machine setting; no action |
| RAM | 62 GiB usable; 53 GiB available at inspection time | Qwen 122B `--no-mmap` is now plausible; the largest MoE models still need evictable cache |
| Swap policy | swappiness 150; 4.1 GiB swap already occupied at inspection time | Record swap deltas during every RAM-heavy experiment |
| NUMA | One node | Do not spend time on NUMA flags |

Primary hardware references: [Intel 265KF specifications](https://www.intel.com/content/www/us/en/products/sku/241062/intel-core-ultra-7-processor-265kf-30m-cache-up-to-5-50-ghz/specifications.html) and [MSI Z890 Carbon specifications](https://us.msi.com/Motherboard/MPG-Z890-CARBON-WIFI/Specification).

## What is already implemented

Do not re-test these generically unless a model shows the matching symptom:

- `-np 1` on the interactive llama.cpp servers.
- q8 K/V cache on most long-context text profiles, with intentional f16 or architecture-specific exceptions.
- Flash Attention enabled on established profiles and `auto` on the new DeepSeek architecture.
- MTP on the Qwen and Gemma profiles where the GGUF actually contains a draft/next-token head.
- Per-model batch and micro-batch choices, including smaller micro-batches where long-context compute buffers are the constraint.
- `--fit on` for compatible layer-split and MoE profiles.
- `--no-mmap` and `--mlock` on models where up-front RAM allocation is safe.
- Explicit GPU UUID ordering for profiles whose balance depends on device order.
- A measured layer-split versus tensor-split comparison; tensor mode is already a specialised, working path here.
- N-gram speculation was tested and removed as a no-op for the active single-slot Qwen workload.

## Guardrails before experiments

### Freeze the software under test

The configs name rolling image tags while comments name known-good llama.cpp builds. Before changing flags:

1. Record `llama-swap --version`.
2. Record `/app/llama-server --version` for each image family.
3. Record each Docker image ID and repository digest.
4. Save the exact expanded server command and relevant environment variables with every result.
5. Do not run `update.sh` until the current TP image is identified and recoverably pinned.

Why this is P0: current upstream [multi-GPU documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md) says tensor mode requires non-quantised KV and does not support auto-fit. `Qwen3.6-27B-Q5-MTP-TP` correctly disables fit but currently uses q8 K/V. That may be valid on its known-good build, but a moving image can turn it into a startup error.

### Use one benchmark contract

For every variant, run the same four cases:

1. Cold request after model load.
2. Repeated warm request with cache reuse.
3. A representative coding/tool loop.
4. The existing 31K NIAH prompt, plus a near-maximum-context prefill for profiles intended to serve more than 64K.

Use a fixed seed and output-token budget. Run at least three warm repetitions; compare medians and note range. Capture:

- model-load time, TTFT, PP, TG and full request time;
- MTP accepted tokens or acceptance ratio;
- per-GPU VRAM and utilisation at load, after prefill and after generation;
- CPU utilisation by CPU ID and effective frequency;
- RAM, page-cache and `pswpin`/`pswpout` deltas;
- logs for OOM, graph recapture, page faults, NCCL warnings and fallback kernels.

Keep a change only if it improves the intended metric by at least about **5%**, improves a material tail-latency/stability problem, or unlocks useful context/quality. Reject it if quality changes, swap grows continuously, a GPU falls off the bus, or a long-context prefill OOMs.

## Prioritised experiments

### P1 — Pin RAM-offloaded MoE generation to the eight P-cores

**Targets:** `qwen3.5-122B-IQ3-MTP`, `laguna-s-2.1-mainline`, and, once it has a clean baseline, `deepseek-v4-flash-IQ2`.

**Qwen3.5-122B result, 02/08/2026: rejected.** Tested on llama.cpp b10143 with the same model, placement, 128K context, MTP, KV and batch settings. The variant changed `--threads 12` with unrestricted CPUs 0–19 to eight threads strictly confined to P-core CPUs 0–7.

| Metric | Existing profile | P-core-only | Change |
| --- | ---: | ---: | ---: |
| 29,361-token prompt processing | 148.78 t/s | 151.40 t/s | +1.8% |
| Warm 256-token generation | 29.92 t/s | 25.08 t/s | **-16.2%** |
| Warm MTP acceptance | 135/239 | 135/239 | unchanged |

The small PP gain does not compensate for the large user-visible TG regression. Keep Qwen3.5-122B at `--threads 12` with normal scheduler placement. The unchanged acceptance indicates that the slowdown came from restricting CPU execution capacity, not different speculative behaviour. Do not generalise the source guide's i5-12600K result to this Arrow Lake CPU. Laguna and DeepSeek remain separate model-specific questions, not reasons to retain this rejected Qwen variant.

These profiles execute CPU-resident experts and currently request 12 or 20 unpinned threads. On this CPU, CPUs 0–7 are the 5.4–5.5 GHz P-cores and 8–19 are the 4.6 GHz E-cores.

Compare:

| Variant | Generation | Prompt processing |
| --- | --- | --- |
| A | Current flags | Current flags |
| B | `--threads 8 -Cr 0-7 --cpu-strict 1` | `--threads-batch 8 -Crb 0-7 --cpu-strict-batch 1` |
| C, only if B helps TG but hurts PP | P-cores as in B | All 20 cores via `--threads-batch 20 -Crb 0-19` |

Use llama.cpp affinity flags first so generation and batch affinity can differ. If B is the winner, optionally enforce the final boundary with Docker `--cpuset-cpus=0-7`.

Success: higher TG or lower TTFT variance without a material PP regression. Do not apply this to fully GPU-resident dense models without evidence.

### P1 — Evaluate DeepSeek V4 DSpark at 131K

**Result, 03/08/2026: promising opt-in; do not replace the 262K profile yet.** The verified 10.14 GiB DSpark GGUF loaded successfully on the unified CUDA image at digest `sha256:97ecc6acdc6341031d3f4f0cf48514ec735c1ff0c08c29f66628deda3c2dbe3d` (llama.cpp commit `2b63e06`). The separate `deepseek-v4-flash-IQ2-dspark-131K` llama-swap entry passed an end-to-end proxied request.

The working placement is intentionally explicit:

- `--fit off`: automatic fit fails to measure the `dflash` context and then hits `GGML_SCHED_MAX_SPLIT_INPUTS`.
- `--cpu-moe`: target MoE tensors stay in evictable mmap/CPU memory.
- `--device CUDA0,CUDA1,CUDA2 --tensor-split 0.1,1,1`: most target dense weights go to the two 5060 Ti cards.
- `--override-tensor output.weight=CUDA0`: DSpark reuses this target tensor, so it must share the 5070 Ti with the draft.
- `--spec-draft-device CUDA0 --gpu-layers-draft 4`: the entire draft stays on one GPU; splitting it triggered early scheduler/device failures.
- Use `--gpu-layers`, not the advertised `--ngl` alias, which is broken in this image.

Fixed prompt and seed, temperature 1.0, 128 generated tokens, one cold probe followed by three warm trials:

| Configuration | Warm TG | Draft acceptance | Interpretation |
| --- | ---: | ---: | --- |
| 131K auto-fit, no draft | 12.46, 13.36, 13.84 t/s | — | Existing-placement reference; median **13.36** |
| Exact DSpark target placement, no draft | 7.90, 9.07, 9.00 t/s | — | Placement-matched control; median **9.00** |
| DSpark, first full pass | 7.02 t/s | 85/122 (69.7%) | Draft weight pages still warming |
| DSpark, fully warm | 17.02, 17.53 t/s | 85/122 (69.7%) | Median **17.28** across the stable pair |

Using all three DSpark trials gives a conservative 17.02 t/s median: **+89% versus the placement-matched control and +27% versus the 13.36 t/s auto-fit reference**. The loaded speculative slot reported 131,072 context. Warm DSpark left 11.8/1.7/12.1 GiB free on host NVIDIA indices 0/1/2 respectively, with 31.8 GiB container RSS and about 54 GiB system RAM still available.

Correctness caveat: output was coherent and repeated DSpark runs were deterministic, but a greedy 128-token response was not bit-identical to the no-draft control even after matching prompt-cache state. This may be numerical/batching sensitivity rather than semantic degradation, but it means the current WIP implementation has not earned a strict losslessness claim on this setup.

Before promotion, run the existing 31K NIAH test, a near-131K prefill, multiple prompt classes, and a longer soak while recording page faults and swap I/O. Also compare `n_max` 2 versus 3; keep 3 only if its acceptance and end-to-end gain survive broader prompts. Upstream's merge notes describe DSpark as WIP and report a 46.4% aggregate acceptance across a broader nine-prompt benchmark, so the measured 69.7% on one coding prompt should not be generalised.

### P1 — A/B the host power profile

The observed `powersave` governor and `balance_performance` EPP make this worth measuring on a CPU-heavy MoE profile.

1. Benchmark the current profile after a warm-up.
2. Switch temporarily to the desktop's performance profile and confirm governor/EPP/frequency again.
3. Repeat the identical benchmark.
4. Switch back before testing another variable.

Only consider replacing `power-profiles-daemon` with `tuned-ppd` if the reversible performance-profile test shows a repeatable loss or boot-to-boot variance. A permanent throughput profile increases idle power and heat and should earn its place with measurements.

Success: at least 5% TG/PP improvement or materially lower run-to-run variation on a RAM-offloaded MoE model.

### P1 — Add a test-only `--no-mmap` variant for Qwen3.5-122B

The config comments already identify this as the main benefit of the 64 GB RAM upgrade: earlier measurement improved 32K prefill from 277 to 600 t/s, while 32 GB made the 128K profile unsafe. The active command still uses mmap.

Compare the current 128K profile with the same command plus `--no-mmap`:

- run the 31K NIAH test;
- run a near-128K prefill;
- watch resident RAM, available RAM and swap deltas through a 30–60 minute session;
- leave `--mlock` out initially so this experiment tests mapping, not two memory policies at once.

Success: a large PP improvement, no more than a small TG regression, no sustained swap growth, and enough RAM for the desktop and persistent dictation group. If it swaps or threatens the driver watchdog, retain mmap.

### P1 — Protect the working tensor-parallel profile from upstream drift

For `Qwen3.6-27B-Q5-MTP-TP`:

1. Capture and pin the exact working llama.cpp image and NCCL 2.30.4 combination.
2. Re-run the existing q8-KV profile and save output-quality checks, PP, TG and 175K prefill stability.
3. Build a separate current-upstream variant using f16 K/V, as upstream now requires; lower context as needed to fit.
4. Compare speed, context ceiling, VRAM and output quality.
5. Update production only if the new combination wins or fixes a real fault.

Do not add `GGML_CUDA_P2P=1` casually. The host has a history of multi-GPU stability problems, the current path deliberately uses NCCL SHM with P2P disabled, and upstream warns that consumer P2P can crash or corrupt output.

### P2 — Fit-headroom and micro-batch sweep for Qwen3.6-35B

`qwen3.6-35B-Q4` uses `--fit-target 256` and `-b 2048 -ub 2048`. The guide's persistent-server floor is 512 MiB, and a full-size micro-batch maximises the compute buffer.

Run this small matrix at the configured 128K context:

| Fit target | Micro-batch |
| ---: | ---: |
| 256 | 2048 — current baseline |
| 512 | 2048 |
| 512 | 1024 |
| 512 | 512 |

Measure initial placement, PP, TG, VRAM after a near-128K prefill, and a long-session soak. Prefer the fastest variant that survives the real context with comfortable margin; do not reserve headroom merely to match someone else's 12 GB card.

### P2 — Target/draft KV precision on Gemma MTP

Start with `gemma-4-31B-Q4-MTP`, where the current q8 target KV produces roughly 61–63% draft acceptance.

Compare:

1. Current q8 target cache.
2. f16 target K/V with explicit f16 draft K/V.
3. If f16 improves acceptance enough to win end-to-end, sweep draft length 2, 3 and 4.

Record acceptance, TG, full request latency and VRAM. An acceptance gain is not a win if f16 forces too much context or model placement out of VRAM. Repeat on the 26B MoE only if the 31B result indicates that precision is important on these Gemma builds.

### P2 — Verify XMP/configured RAM speed once

Use `dmidecode` or the BIOS to compare configured DIMM speed with the intended kit/XMP profile. This is a validation step, not an invitation to change memory timings blindly. If the current speed is already correct, close the item permanently.

### P3 — CUDA graphs and scheduling priority

Only after the higher-value tests:

- Compare `GGML_CUDA_GRAPH_OPT=0` and `1` through a variable-context 30–60 minute session, not a short decode benchmark.
- Test `--prio 2` for TTFT/TG tail-latency variance while the desktop is busy.
- Treat `--no-warmup` as a startup-latency trade, not a throughput optimisation.

Reject a CUDA-graph variant that grows VRAM or OOMs late even if its short benchmark wins.

## iGPU display trick and gaming

### On the current hardware: it will not work

The installed 265KF has no processor graphics. Intel lists graphics output as unavailable, zero Xe cores and zero displays. MSI marks the board's HDMI and USB-C display outputs as available only with a graphics-enabled CPU. Plugging the monitor into the motherboard will therefore produce no display; there is no BIOS toggle that can create the missing GPU.

This means the proposed change cannot break the 5070 gaming setup—because it cannot be enabled in the first place.

### If the CPU were replaced with a 265K or another iGPU-equipped chip

Gaming on the 5070 would still be possible in principle through render offload, but the existing proven solution would no longer be the same solution:

- `MESA_VK_DEVICE_SELECT=10de:2c05` would still request the 5070 for the game.
- `gamescope --prefer-vk-device 10de:2c05` would still request the 5070 for gamescope.
- KWin and physical scan-out would now be on Intel, so a cross-GPU copy/presentation path would be unavoidable.
- The current gaming note explicitly relies on keeping gamescope and presentation on the 5070 to avoid that copy and to work around a presentation failure.

Therefore an iGPU conversion would require a fresh 1440p/60 and 4K/60 gaming validation: launch reliability, frametime, latency, VRR/HDR if used, suspend/resume, and Ubisoft/Proton behaviour. It is not sensible to buy a replacement CPU merely to reclaim roughly 0.5–1 GB of VRAM.

See [Gaming on the 5070 Ti (multi-GPU)](</home/jim/Documents/obsidian-md/Tech/Home Setup/Marvin/Gaming on the 5070 Ti (multi-GPU).md>).

### Lower-risk alternatives

1. **Keep the 5070 as the display GPU.** This preserves the known-good direct gaming path; treat its desktop VRAM as part of the tensor split and fit budget.
2. **Use an explicit headless inference mode only when a marginal model needs it.** Stopping the graphical target frees compositor VRAM but logs out the desktop and precludes local gaming until graphical mode is restored.
3. **Only if measurements prove 5070 display VRAM is the binding constraint, test a 5060 as the desktop GPU.** For the two-GPU TP pair this could move desktop overhead onto the otherwise separate ZOTAC, but it competes with dictation there; for three-GPU fit profiles it mostly moves the overhead rather than eliminating it. Gaming would need the same cross-GPU revalidation or a physical input switch back to the 5070.

## Explicitly skip for now

- Replacing the CPU for an iGPU: poor value for the amount of VRAM reclaimed.
- Global `--mlock`/`--no-mmap`: unsafe for Laguna and DeepSeek, whose large CPU-side expert sets deliberately rely on evictable page cache.
- NUMA tuning: one NUMA node.
- Poll sweeps: low-value and already flat in the source guide.
- N-gram speculation on the current Qwen single-slot path: already measured as a no-op.
- Blind image updates or a source-build migration: reproducibility and the working custom NCCL/TP path matter more than following the guide's preferred packaging.
- Sampling changes during performance tests: they confound quality without addressing the bottleneck.

## Recommended execution order

1. Capture versions/digests and create the benchmark result format.
2. Establish current baselines for Qwen 122B, Laguna, Qwen 35B, Gemma 31B MTP and Qwen 27B TP.
3. Run P-core affinity and power-profile experiments independently.
4. Test Qwen 122B `--no-mmap`.
5. Pin and audit the TP runtime before any update.
6. Run Qwen 35B fit/micro-batch and Gemma KV-precision sweeps.
7. Only then try CUDA graphs, priority or display-path changes.
8. Promote winners one at a time, with the result and exact command recorded beside the config.

## Confidence and gaps

High confidence: the current CPU has no iGPU; CPUs 0–7 are the P-cores; the monitor is connected to the 5070; the current configuration already implements most guide recommendations; upstream tensor mode now documents non-quantised KV and no fit.

Medium confidence until benchmarked: P-core pinning and a performance power profile will help these particular Arrow Lake MoE runs; Qwen 122B `--no-mmap` will remain safe during a full 128K session on the current desktop workload.

Not established from the sandbox: current per-GPU VRAM usage and active image digests, because NVIDIA and Docker device access were unavailable to the inspection process. Capture both on the host as the first execution step.
