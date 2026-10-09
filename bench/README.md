# Hardware benchmark: how to repeat it

These scripts produced the "before" numbers in
[`../hardware-upgrade-benchmarks.md`](../hardware-upgrade-benchmarks.md) (2026-10-09,
Core Ultra 7 265KF / Z890 / 64 GB DDR5-6000). Run them unchanged on the new platform
(EPYC 7443 / H12SSL-NT / 256 GB DDR4-2666) and add an "after" column to that document.

Total run time is about 25 minutes.

## Rules for a fair comparison

Change only the hardware. Anything else that changes must be written down next to the
results.

- **Same software:**
  - llama.cpp image `ghcr.io/ggml-org/llama.cpp:server-cuda13` at **build 11515**
    (`commit 3d65c90d0`).
  - The Strata marvin fork at commit **`d7b05f7`** (branch `marvin-tuned-0.1.41`).
  - ComfyUI `b65d1ffa`.
  - The llama-swap `config.yaml` entries as they were in commit `8da268c`.

  The llama.cpp image tag is a rolling tag, so it will have moved on. Pull the same
  build by its digest or tag (`server-cuda13-b11515` if it is published), or record
  the new build next to the results.
- **Same GPU placement:** the configs pin GPUs by UUID, so they follow the cards to
  the new slots. Keep the NVIDIA driver at 615.71.09 if possible, and keep the power
  limits: 3090 280 W, 5070 Ti 260 W, 5060 Ti 150 W each.
- **Idle machine:** check `uptime` and `top`. On 2026-10-09 an orphaned Playwright
  `chrome-headless-shell` was burning 15 cores and had to be killed first.
- **Nothing else on the GPUs:**
  - Unload Strata (`curl -X POST localhost:8033/api/models/unload/Strata-IQ3S`).
  - Free ComfyUI (`curl -X POST localhost:8190/free -H 'Content-Type: application/json' -d '{"unload_models":true,"free_memory":true}'`).
  - The voice stack on the ZOTAC/NPU stayed loaded in the baseline: s1-mini and
    chatterbox, about 3.4 GB on the ZOTAC.
  - **The NPU will not exist on the EPYC** (it is part of the Core Ultra), so
    whisper-npu-asr will not run there.
- **Unchanged input files:** `test-30K.md` (sha256 `0280e0223ac0ac62...`) and the
  scripts in this folder. `sha256sum test-30K.md bench/*` must match the values in the
  results document.

## 1. RAM bandwidth (about 1 minute)

```sh
cd ~/llama-swap/bench
gcc -O3 -march=native -fopenmp -o mem_bw mem_bw.c
OMP_NUM_THREADS=$(nproc) OMP_PROC_BIND=spread ./mem_bw 2048
OMP_NUM_THREADS=8 OMP_PROC_BIND=spread ./mem_bw 2048
OMP_NUM_THREADS=1 ./mem_bw 2048
```

`mem_bw.c` is a STREAM-style test: three 2 GiB arrays and the best of 10 passes, with
STREAM's byte counting. The baseline used 20 threads (all cores); on the EPYC, also
record 48 threads (`nproc`) and 24. Compare **triad**.

## 2. PCIe host<->GPU bandwidth (about 30 seconds)

```sh
~/comfyui/ComfyUI/.venv/bin/python ~/llama-swap/bench/pcie_bw.py
```

This copies 256 MiB of pinned memory to and from each GPU (best of 20). It also prints
the negotiated link, read while the copy is running. Check that each card shows the
expected `16.0 GT/s x8` or `x16`.

## 3. SSD throughput (about 3 minutes)

```sh
~/llama-swap/bench/disk.sh /home/jim /mnt/models
```

This runs fio with O_DIRECT on a temporary 4 GiB file per drive:
- sequential read and write, 1 MiB blocks, queue depth 32;
- random 4K read at queue depth 32 × 4 jobs;
- random 4K read at queue depth 1.

`/home` is the WD SN7100 (btrfs). `/mnt/models` is the KIOXIA EXCERIA BASIC (xfs).
Both are on chipset Gen4 x4 links today. If they move to CPU-attached M.2 slots, note
it.

## 4. Images in ComfyUI (about 3 minutes)

```sh
python3 ~/llama-swap/bench/image_bench.py
```

For each model the script first unloads everything in ComfyUI (`POST /free`), then
renders two images:
- **1024×1024, cold:** includes loading the models from disk.
- **1920×1088, warm:** the models are already loaded.

The graphs are built inside the script.

### FLUX.2 Klein 9B

This is `~/comfyui/workflows/flux2-klein-9b.json`.

| Setting | Value |
|---|---|
| Diffusion model | `Flux2-Klein-9B-True-v2-Q8_0.gguf` (UnetLoaderGGUF) on the MSI |
| Text encoder | `qwen_3_8b_fp4mixed.safetensors` on the ZOTAC (`SelectCLIPDevice gpu:1`) |
| VAE | `flux2_ae.safetensors` |
| Sampling | 8 steps, euler, Flux2Scheduler, CFG 1 |
| Seed | `20261009` |

Prompt:

> A lighthouse on a rocky headland at dusk, painted in bold flat gouache shapes, limited palette of teal, rust and cream, visible brush texture

### Qwen Image 2.1

This is the official "Qwen Image 2.1: Text to Image" template with the prompt enhancer
off, the same graph as the `qwen-eval` runs.

| Setting | Value |
|---|---|
| Diffusion model | `qwen_image_2.1_int8_convrot.safetensors` through QwenImage21Cache (auto) |
| Text encoder | `qwen3vl_8b_int8_convrot.safetensors` |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` |
| Text encode | `resolution` 1024, empty negative prompt |
| Sampling | 25 steps, euler, simple, CFG 1 |
| Seed | `20261009` |

Prompt:

> A quiet harbour town at dawn seen from a hillside: terracotta roofs, a stone breakwater with a small red lighthouse, fishing boats at their moorings, mist lying on the water, warm low sunlight, detailed realistic photograph

The images are saved as `ComfyUI/output/hwbench-*`. Compare them by eye with the
baseline files: different hardware can change pixels slightly, but not the picture.

## 5. LLMs (about 15 minutes)

```sh
cd ~/llama-swap
python3 bench/llm_bench.py Strata-IQ3S
curl -X POST localhost:8033/api/models/unload/Strata-IQ3S
python3 bench/llm_bench.py Q3.8-27B-IQ4XS Q3.8-27B-Q6KM Q3.8-27B-IQ4XS-DF2 \
    muse-glimmer-30B-dflash-vision gemma-4-31B-Q4-MTP
```

Requests go through llama-swap (`127.0.0.1:8033`), so each model runs exactly as
configured. For each model the script runs:

1. **Load:** "Reply with OK." with `max_tokens` 8. The time includes starting the
   model. In the baseline the GGUFs were probably in the page cache, so this is not a
   cold-disk load time.
2. **Short prompt, long generation:** `max_tokens` 2048, `seed` 42, streamed. Prompt:

   > Write a detailed technical essay of at least 3,000 words on how a large city's water supply works, from reservoir to tap: sources, treatment stages, pumping, storage, distribution networks, pressure management, leak detection and maintenance. Use headed sections and full paragraphs.

3. **Long prompt:** the full text of `~/llama-swap/test-30K.md` (about 29.4K tokens),
   then two newlines and:

   > In about 300 words, summarise the main events of this chronicle.

   `max_tokens` 512, `seed` 42.

Every prompt starts with a line `[run <random 12 hex digits>]`, so no prefix cache is
reused between runs. Sampling and thinking settings are each model's own; the script
overrides nothing except `seed` and `max_tokens`.

Speeds come from the server's own `timings`, which both llama-server and Strata return.

| Metric | What it measures |
|---|---|
| `prefill_tps` | Prompt tokens processed per second |
| `decode_tps` | Generated tokens per second, including speculative-decoding gains |
| `draft_accept` | The share of drafted tokens accepted |

Results are appended to `bench/results/llm.jsonl` and `bench/results/image.jsonl`.

Afterwards, reload production with `curl localhost:8033/v1/chat/completions -d
'{"model":"Strata-IQ3S","messages":[{"role":"user","content":"hi"}],"max_tokens":4}'`.
