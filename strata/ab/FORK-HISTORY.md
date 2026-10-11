# marvin-tuned: Strata for Marvin

This fork is tuned for one machine:

- **CPU:** Intel Core Ultra 7 265KF (8 P-cores and 12 E-cores, AVX2 only, no AVX-512).
- **RAM:** 64 GB, headless (no desktop running).
- **OS:** Linux (CachyOS) with CUDA 13.4.
- **GPUs:** an RTX 5070 Ti 16 GB (sm_120) and an RTX 3090 24 GB (sm_86), both on PCIe gen 4 x8, split by layer (`"gpu": [2, 1]`).
- **Model:** Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, 262K context, int8 KV with KV streaming.

The fork is based on upstream 0.1.40 (rebased 2026-10-06; the 0.1.37 version is tag `marvin-tuned-0.1.37`).

## What the fork changes

Upstream 0.1.40 (#848, "Resident RAM mode with a layer split") now does what this fork first added on 0.1.37:

- the pinned resident RAM tier works across a layer split;
- it is ranked by the whole expert profile;
- adaptive swaps copy back from the card that owns the layer.

Those parts were dropped in the rebase. The fork has one code change left, in `src/program/generate.cpp` and `src/core/expert_source.cpp`:

- **Each card's prompt loan stays in RAM.** These are the top cache slots each card lends to the prompt path's buffers. On a split, upstream turns the loan's RAM copy off, so a prompt reads those experts from the files and refills them from the files after. The fork keeps them in the pinned copy (`keep_in_ram`), so a prompt streams and refills them by DMA. Set `STRATA_RESIDENT_LENT=0` to turn it off.

The 0.1.37 fork's opt-in THP backing (`STRATA_COMPLEMENT_THP`) was dropped: it never paid off on this box.

## Fork vs mainline 0.1.40 (2026-10-06, IQ3_XXS, through llama-swap)

Mainline is what 0.1.40's setup writes for this box: the official config with `--resident-experts` (it pins the 11.9 GiB of experts no card holds). The fork is `strata-iq3_xxs-marvin.json`. Same harness as below (`marvin/ab/bench.py`), 2 reps, medians, tok/s. Raw results are in `marvin/ab040/` (mainline ran from `~/Strata` with `engine/strata` built from the tag).

| | mainline 0.1.40 | **fork** | |
|---|---|---|---|
| 1K / 4K prompt | 430 / 1281 | **734 / 1728** | +71% / +35% |
| 16K / 32K prompt | 2217 / 2717 | **2885 / 3516** | +30% / +29% |
| 100K prompt | 3223 | **4062** | +26% |
| decode: code / prose / explain | 119 / 91 / 103 | **132 / 106 / 116** | +11% / +17% / +13% |
| agent (17.8K prompt): prompt / decode | 1746 / 99 | **2644 / 113** | +51% / +14% |
| pinned RAM | 11.9 GiB | 19.2 GiB | |

The benchmark prompts come from the `~/Strata` source, which is now 0.1.40, so these numbers are not comparable with the 0.1.37 tables below.

## History: the 0.1.37 fork

On 0.1.37, a two-GPU layer split gave two choices:

- **The arena** (no `--mmap-experts`). It pins every expert, about 44 GiB. It's fast, but it leaves the machine about 4 GiB of available RAM.
- **`--mmap-experts`**, with the pinned resident tier refused on a split. Nothing was page-locked, so the CPU computed every miss and prompts copied every streamed expert through host threads.

The 0.1.37 fork made the resident tier work across the split, which upstream then did independently in 0.1.40.

## The settings that matter (`strata-iq3_xxs-marvin.json`)

- `--mmap-experts --resident-budget-gib 22`: about 19.2 GiB pinned (the uncached experts plus both cards' prompt loans). About 25 GiB of RAM stays available.
- `--pool-workers 8` instead of upstream's default of every core but one (19 here). Measured: 8 to 14 workers perform the same once the misses are pinned, and 19 was 5–9% slower.
- Unchanged, because the sweeps below confirmed them: `--spec 4 --spec-min-p 0.5`, the probed `--pcie-frac`, and an 8K prompt chunk.

## Measured on 0.1.37 (2026-10-03, through llama-swap)

Method:

- Unique nonce on every prompt, temperature 0, 2 reps, medians.
- Prompt figures are tok/s reading the prompt. Decode figures are tok/s for 1024 generated tokens.
- `agent` is a 17K-token prompt followed by 700 generated tokens.

| | official 0.1.37 (mmap) | official, arena | **marvin-tuned** |
|---|---|---|---|
| 1K / 4K prompt | 300 / 960 | 547 / 1583 | **645 / 1601** |
| 16K / 32K prompt | 1822 / 2169 | 2165 / 3018 | **2625 / 3373** |
| 100K prompt | 2389 | **4117** | 3795 |
| decode: code / prose / explain | 97 / 75 / 87 | 118 / 90 / 102 | 115 / 90 / 101 |
| agent: prompt / decode | 1701 / 88 | 2027 / 97 | **2546 / 98** |
| RAM available | ~45 GiB | ~4 GiB | ~25 GiB |

Things that did not help:

- `--spec 5` or `6`, and `--spec-min-p 0.3` or `0.7`: equal or worse.
- `--pcie-frac 0.55` instead of the probed 0.35–0.37: worse.
- A 16K prompt chunk with the resident tier: 16K, 32K and agent prompts were about 20% slower (100K was 3% faster).
- A VRAM reserve of 350 MiB instead of 700: the verify buffers no longer fit, so the engine failed to start.

Raw results and the harness are in `marvin/ab` (`bench.py`, `sweep.py`, `results/`). The benchmark prompts are built from the upstream 0.1.37 source in `~/Strata`, so keep that checkout at the same version when comparing new runs with these.

## IQ3_S on the fork (`strata-iq3_s-marvin.json`)

The bigger quant (setup calls it "matches the full model") fits with the fork:

- `--resident-budget-gib 30` pins 27.8 GiB, and about 16 GiB of RAM stays available.
- `--vram-reserve-mib 957`, which is what the engine asked for. Without it only 255 MiB of VRAM was free.
- Shard 2, the PLE/engram table, is the same file (same SHA-256) for every GSQ-RCO size. Only shard 1 (54.8 GB) was downloaded; shard 2 is a symlink to the IQ3_XXS copy.

Measured with the same harness (tok/s):

| | IQ3_XXS fork | IQ3_S fork |
|---|---|---|
| 4K / 16K / 32K / 100K prompt | 1601 / 2625 / 3373 / 3795 | 1354 / 2399 / 3003 / 3326 |
| decode: code / prose / explain | 115 / 90 / 101 | 99 / 80 / 91 |

IQ3_S is about 10-15% slower than IQ3_XXS on the fork, and about level with official 0.1.37 on IQ3_XXS.

llama-swap runs it as the `Strata-IQ3S` entry (`~/lab/Strata/run-strata-qwen-iq3s.sh`). It cannot run beside Strata-IQ3XXS: both use the same two cards.

## Pipelined windows on, parallel slots off (2026-10-06, IQ3_S, fork on 0.1.40)

Both configs run `--pipeline-windows 2`: the 5070 Ti starts the next verify window while the 3090 still finishes this one, on the guess that the drafts are accepted. Measured one request at a time (tok/s, medians of 2):

| | without | `--pipeline-windows 2` | `"parallel": 2` |
|---|---|---|---|
| code decode | 108 | **135 (+25%)** | 104 |
| prose decode | 102 | 103 | 98 |
| 30K prompt | 2874 | 2800 | 2772 |

`--pipeline-windows 2` keeps 278 MiB of the 5070 Ti out of the expert cache. Code gains most because its drafts are guessed right more often.

Parallel slots (`"parallel": 2 --batch-groups 2`) were measured and left off:

- Each slot takes about 0.5 GiB on each card from the expert cache, which costs about 4% on every request.
- Two requests sent together finished at 23.8 s and 23.4 s, with 86 tok/s combined. Queued one after the other, they finished at 10.2 s and 20.7 s, with 97 tok/s combined. The cause is that slots on a split decode without MTP drafts.
- The two options can't be combined anyway.

## Remote access: `allowed_hosts`

From 0.1.40 the server rejects any request whose Host header is not a name it knows (DNS-rebinding protection, 403 "Host ... is not allowed"). llama-swap passes on the client's Host, so every request over the tailnet (`marvin.akita-betelgeuse.ts.net:8033`) was refused until both configs got `"allowed_hosts": [".akita-betelgeuse.ts.net", "marvin"]`. Keep it in any new config.

## Building

```sh
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  "-DCMAKE_CUDA_ARCHITECTURES=86;120" -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  -DSTRATA_GGML_DIR=/home/jim/lab/Strata/third_party/llama.cpp
cmake --build build --target strata -j 14
cp build/strata engine/strata.new && mv engine/strata.new engine/strata
```

The launchers live in `marvin/` and are symlinked from `~/Strata`:

- `run-strata-qwen.sh` takes an optional engine config as its 2nd argument and waits for any previous engine (`STRATA_WAIT_S`, default 60 s).
- `run-strata-qwen-iq3s.sh` unloads Strata-IQ3XXS once it is idle, then starts IQ3_S.
- `variant.env` is the production switch.

The A/B harness, configs and raw results are in `marvin/ab`.

`~/lab/Strata/run-strata-qwen.sh` runs this fork when `~/lab/Strata/variant.env` points at it. Delete `variant.env` to go back to the official checkout and its config.

To move to a new upstream (on 0.1.40, upstream had absorbed most of the fork, so it was rebuilt on the tag instead of rebased):

```sh
git -C ~/lab/Strata fetch --tags
git rebase <tag>
```

Then rebuild, and re-run the A/B before trusting the numbers.
