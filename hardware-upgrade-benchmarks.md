# Marvin hardware upgrade: benchmarks

Baseline measured on **2026-10-09**, before the move to the EPYC platform. To repeat the
tests, follow [`bench/README.md`](bench/README.md): same scripts, same prompts and the
same seeds. Add the "after" numbers as a new column.

## Hardware

| | Before (2026-10-09) | After (planned) |
|---|---|---|
| CPU | Intel Core Ultra 7 265KF, 20 cores / 20 threads, up to 5.5 GHz (has an NPU) | AMD EPYC 7443, 24 cores / 48 threads, 128 PCIe 4.0 lanes (no NPU) |
| Board | MSI MPG Z890 CARBON WIFI (MS-7E17), BIOS 1.AB0 (2026-06-26) | Supermicro H12SSL-NT (2 × 10 GbE) |
| RAM | 64 GB: 4 × 16 GB DDR5-6000 (Corsair CMK32GX5M2E6000Z36 + CMK5X16G1E60Z36A2), dual channel, 96 GB/s theoretical | 256 GB DDR4-2666 2Rx4 RDIMM, 8 channels, 170 GB/s theoretical |
| GPU 0 | MSI RTX 5060 Ti 16 GB, CPU M.2-adapter slot, Gen4 **x4**, 150 W | Gen4 x8 |
| GPU 1 | RTX 3090 24 GB, CPU slot, Gen4 **x8**, 280 W | Gen4 x16 |
| GPU 2 | RTX 5070 Ti 16 GB, CPU slot, Gen4 **x8**, 260 W | Gen4 x16 |
| GPU 3 | ZOTAC RTX 5060 Ti 16 GB, **chipset** slot, Gen4 **x4**, 150 W | Gen4 x8 |
| SSD 1 | WD_BLACK SN7100 1 TB (fw 7615M0WD), chipset Gen4 x4, btrfs: `/`, `/home` | |
| SSD 2 | KIOXIA EXCERIA BASIC 1 TB (fw A1RA0104), chipset Gen4 x4, xfs `/mnt/models`, `/mnt/shared` | |

The 5060 Tis and the 5070 Ti are Gen5 cards, but every slot today is Gen4, and so are
the H12SSL's. So the gain comes from link width, not link speed.

## Software

| | Version |
|---|---|
| OS | CachyOS, kernel 7.2.7-1-cachyos, THP `always` |
| NVIDIA driver / CUDA toolkit | 615.71.09 / nvcc 13.4.92 |
| Docker | 29.8.1 |
| llama-swap | v258 (69cb75a); config at repo commit `8da268c` |
| llama.cpp | `ghcr.io/ggml-org/llama.cpp:server-cuda13`, build 11515 (commit 3d65c90d0), pulled 2026-10-09 |
| Strata | marvin-tuned fork on upstream **v0.1.41**: `~/Strata-marvin`, branch `marvin-tuned-0.1.41`, commit `d7b05f7`, engine built 2026-10-09 (sm 86;120) |
| ComfyUI | 0.38.0 (`b65d1ffa`), PyTorch 2.14.1+cu130, ComfyUI-GGUF `6ea2651` |
| Input checksums (sha256, first 16) | `test-30K.md` 0280e0223ac0ac62 · `llm_bench.py` 513606232264132f (0a0601696d414146 before the unload step was added) · `image_bench.py` d6dc5cf1faaaad02 · `disk.sh` 783c64869c076d8b · `mem_bw.c` c3c5fe58ac6ec3f9 · `pcie_bw.py` 42b19046af083b9c |

Conditions:
- The machine was otherwise idle. An orphaned headless Chromium burning 15 cores since
  2026-10-01 was killed first.
- Strata and ComfyUI were unloaded except during their own tests.
- The voice stack stayed loaded on the ZOTAC: s1-mini and chatterbox, about 3.4 GB,
  plus whisper on the NPU.

## RAM bandwidth (`bench/mem_bw.c`, STREAM-style, GB/s)

| Threads | Copy | Scale | Add | Triad |
|---|---|---|---|---|
| 20 (all cores) | 53.8 | 53.7 | 60.4 | **58.8** |
| 8 | 51.8 | 51.7 | 57.8 | 56.9 |
| 1 | 28.3 | 28.2 | 31.9 | 32.3 |

Triad reaches about 61% of the 96 GB/s theoretical. That share is typical for client
DDR5 measured with STREAM's byte counting.

## PCIe host↔GPU bandwidth (`bench/pcie_bw.py`, pinned memory, GB/s)

| GPU | Link while copying | Host→GPU | GPU→host |
|---|---|---|---|
| 0 MSI 5060 Ti | 16 GT/s x4 | 6.4 | 7.1 |
| 1 RTX 3090 | 16 GT/s x8 | 12.8 | 13.2 |
| 2 RTX 5070 Ti | 16 GT/s x8 | 13.6 | 14.3 |
| 3 ZOTAC 5060 Ti | 16 GT/s x4 (via chipset) | 7.1 | 7.1 |

Expected after the upgrade: about 13–14 GB/s on the 5060 Tis and about 25 GB/s on the
3090 and 5070 Ti.

## SSD (`bench/disk.sh`, fio O_DIRECT, 4 GiB file)

| Drive | Seq read 1M QD32 | Seq write 1M QD32 | Rand 4K read QD32×4 | Rand 4K read QD1 |
|---|---|---|---|---|
| WD SN7100 (`/home`, btrfs) | 7097 MB/s | 2622 MB/s | 418 MB/s · 102K IOPS | 46 MB/s · 11.2K IOPS (83 µs) |
| KIOXIA (`/mnt/models`, xfs) | 7085 MB/s | 6582 MB/s | 4038 MB/s · 986K IOPS | 88 MB/s · 21.4K IOPS (44 µs) |

The WD's weaker write and random results are probably btrfs (checksums and
copy-on-write under O_DIRECT) more than the drive itself.

## Images (ComfyUI, `bench/image_bench.py`)

Time is ComfyUI's own execution time per image. The first image of each model is cold:
ComfyUI unloaded everything first, so it includes loading the models.

| Model | 1024×1024 (cold) | 1920×1088 (warm) |
|---|---|---|
| FLUX.2 Klein 9B Q8 GGUF, 8 steps (MSI + encoder on ZOTAC) | 38.1 s | 33.3 s |
| Qwen Image 2.1 int8, 25 steps | 25.2 s | 43.9 s |

## LLMs (`bench/llm_bench.py`, through llama-swap)

The speeds are the server's own figures, in tokens per second. The first five rows
were measured at 21:04–21:10 and the last four at 22:15–22:23. The DF2 entry and the
second gemma-4-E4B run used the version of `llm_bench.py` that unloads other models
first; for the earlier rows the other models were unloaded by hand.

| Model | GPUs | Load | Short prompt → 2048 tok: decode (draft accept) | 29K prompt: prefill / time to first token | 29K prompt: decode (draft accept) |
|---|---|---|---|---|---|
| Strata-IQ3S (0.1.41 fork) | 5070 Ti + 3090, experts in RAM | 44.3 s | **113.3** (56%) | **2771** / 10.6 s | **119.2** (80%) |
| Q3.8-27B-IQ4XS (-sm tensor, MTP) | 3090 + 5070 Ti | 12.1 s | 74.2 (43%) | 1332 / 22.5 s | 83.6 (60%) |
| Q3.8-27B-Q6KM (-sm tensor, MTP) | 3090 + 5070 Ti | 19.2 s | 63.0 (41%) | 1281 / 23.2 s | 75.5 (46%) |
| muse-glimmer-30B-dflash-vision | 5070 Ti + MSI | 14.0 s | 43.7 (37%) | 1446 / 20.5 s | 68.2 (79%) |
| gemma-4-31B-Q4-MTP | 5070 Ti + MSI | 35.9 s | 45.5 (48%) | 1417 / 20.3 s | 52.9 (89%) |
| Q3.8-27B-IQ4XS-DF2 (layer split, DFlash2 draft) | MSI + 5070 Ti | 6.0 s | 55.2 (33%) | 1288 / 22.9 s | 69.7 (51%) |
| Q3.8-FN-AC-IQ4XS-MTP | 5070 Ti + MSI + 3090 | 25.0 s | 45.9 (38%) | 544 / 54.0 s | 47.0 (41%) |
| gemma-4-26B-MoE-MTP | MSI | 23.5 s | 43.2 (63%) | 532 / 53.8 s | 48.6 (92%) |
| gemma-4-E4B-MTP | ZOTAC (beside the voice stack) | 4.5 s | 114.6 (42%) | 3959 / 7.3 s | 115.7 (83%) |

Notes:
- **Load times** were measured with the model files probably already in the page
  cache, so they are not cold-disk times. Strata's includes building its RAM expert
  tier.
- **The DF2 draft was missing and has been downloaded again.**
  `/mnt/shared/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf` is the Q4_K_M from
  [incoai/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF)
  (1,143,006,816 bytes, sha256 `1a25c56858e1ebe9...`), fetched on 2026-10-09.
- **DF2 is slower than its MTP sibling.** The DFlash2 draft (layer split) decodes at
  55 and 70 t/s, against 74 and 84 t/s for Q3.8-27B-IQ4XS (tensor split, MTP).
- **The Flash-Next and gemma-26B models read long prompts slowly** (about 540 t/s).
- **Decode speed with speculative decoding depends on the content.** A summary of
  provided text drafts well (up to 89% accepted), while the open-ended essay drafts
  poorly. Compare like with like.

## What to watch after the upgrade

- **Prefill on the split models** (Q3.8-27B tensor-parallel, muse, gemma). These move
  data between cards over PCIe, so they should gain the most from x8 → x16.
- **Strata.** Experts stream from RAM over PCIe. It should gain from both the wider
  links and the RAM: 256 GB means the whole expert set can be pinned.
- **Single-thread speed.** It drops: Zen 3 at 4.0 GHz against Arrow Lake at
  5.5 GHz. Watch the 1-thread RAM number and anything bound by the CPU.
- **The whisper-npu-asr model will not run.** The NPU belongs to the Core Ultra CPU,
  so speech-to-text needs a new home: parakeet-asr on a GPU, or whisper on the CPU.
