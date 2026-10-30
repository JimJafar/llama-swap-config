# Strata on Marvin

[Strata](https://github.com/Niko1221/Strata) serves Qwen3.8-Flash-Next GSQ-RCO IQ3_S across the RTX 5070 Ti and RTX 3090, with the experts that don't fit in VRAM in RAM. llama-swap runs it as `Strata-IQ3S` (preloaded, alias `subagent`). The IQ3_XXS quant (`Strata-IQ3XXS`) was deleted on 2026-10-12.

The official checkout is `~/lab/Strata`: its engine, `serve/server.py`, `.venv`, the expert profile. Marvin's own files live here:

| File | What |
|---|---|
| `run-strata.sh PORT CONFIG [OTHER]` | The launcher llama-swap runs. `OTHER`, optional, is another persistent model on the same cards, unloaded once idle. |
| `strata-iq3_s.json` | The tuned engine config, with its `.shared-settings.json` (defaults for requests that set none: reasoning effort medium, 16K max tokens) |
| `ram-guard.sh` | Unloads Strata if available RAM drops below 2 GB (run by hand: `setsid nohup ~/llama-swap/strata/ram-guard.sh 2048 &`; it does not survive a reboot) |
| `ab/bench.py`, `ab/sweep.py` | The A/B harness, through llama-swap; results in `ab/results/` |

Until 2026-10-12 this was a fork (`JimJafar/Strata`, branches `marvin-tuned*`), which added a pinned RAM tier across the layer split and then kept each card's prompt loan in that RAM copy. Upstream took over the first in 0.1.40 and the second in 0.1.43 (`aea9dc3e`, `51d55a9a`), so the fork was retired; only the configs and the harness remain.

## The tuned settings

What 0.1.43's `setup.sh` writes for this box (`~/lab/Strata/strata-iq3_s.json`) is the arena: every expert no card holds is pinned in RAM, KV streaming off, 13 pool workers. With 64 GB of RAM that leaves the machine almost nothing (setup itself says IQ3_S needs ~62 of the 62 GB), so the configs here are hand-edited instead:

- `--mmap-experts --resident-budget-gib 30`: experts are read from the GGUF in place, and only the budget is pinned (the uncached experts first, then the cards' prompt loans). About 16 GiB of RAM stays available on IQ3_S.
- `--kv-resident 32768`: KV streaming, so the 262K context fits.
- `--max-context 262144`: setup caps lower on 62 GB.
- `--pool-workers 8`: 8 to 14 were measured equal once the misses are pinned, 19 was 5-9% slower (0.1.37). Re-checked against setup's 13 on 0.1.43: see below.
- `--pipeline-windows 2`: the 5070 Ti starts the next verify window while the 3090 finishes this one. Code decode +25%, prose and prompts level (0.1.40). It keeps 278 MiB of the 5070 Ti out of the expert cache.
- `--vram-reserve-mib 957`: what the engine asked for; 350 was too little for the verify buffers.
- `"fit_max_tokens": true`: a request asking for more than the context has left gets its cap shortened instead of a 400.
- `"allowed_hosts": [".akita-betelgeuse.ts.net", "marvin"]`: since 0.1.40 the server refuses any Host it doesn't know (DNS-rebinding protection), and llama-swap passes on the client's Host, so every tailnet request got a 403 without it. Keep it in any new config.

Unchanged from setup, because sweeps confirmed them: `--spec 4 --spec-min-p 0.5`, the probed PCIe share, an 8K prompt chunk. Parallel slots (`"parallel": 2`) were measured and left off: each costs ~0.5 GiB per card of expert cache, and two requests together finished later than one after the other (slots on a split decode without MTP drafts).

## Measurements (IQ3_S, tok/s, medians of 2, through llama-swap)

`ab/bench.py`: a unique nonce on every prompt, temperature 0. Prompt cases read the start of a frozen text (`ab/corpus.json.gz`: Strata's own source at 0.1.40, so results stay comparable); decode cases generate 1024 tokens; `agent` is a 17.8K-token prompt then 700 generated.

| | fork 0.1.41 (2026-10-11) | 0.1.43, tuned | 0.1.43, 13 workers |
|---|---|---|---|
| 1K / 4K prompt | 458 / 1407 | 476 / 1432 | 477 / 1418 |
| 16K / 32K prompt | 2460 / 3033 | 2456 / 3073 | 2457 / 3061 |
| decode: code / prose / explain | 149 / 103 / 122 | 144 / 106 / 117 | 147 / 107 / 122 |
| agent: prompt / decode | 2170 / 111 | 2266 / 115 | 2257 / 114 |

0.1.43 with the tuned config matches the fork on 0.1.41 (prompts 0-4% faster, decode within run-to-run noise), so dropping the fork cost nothing. Setup's 13 workers come out level with 8 (decode 1-4% higher, within noise), so the config keeps 8 and leaves the CPU more headroom.

**0.2.7 (2026-10-27), same tuned config, same day as a fresh 0.1.43 run:**

| | 0.1.43 | 0.2.7 | 0.2.7, run 2 | 0.2.7, 13 workers |
|---|---|---|---|---|
| 1K / 4K prompt | 449 / 1439 | 435 / 1391 | 439 / 1467 | 422 / 1380 |
| 16K / 32K prompt | 2443 / 3072 | 2495 / 3156 | 2582 / 3171 | 2499 / 3140 |
| decode: code / prose / explain | 143 / 108 / 119 | 125 / 102 / 111 | 126 / 101 / 108 | 127 / 102 / 108 |
| agent: prompt / decode | 2274 / 116 | 2360 / 103 | 2439 / 103 | 2349 / 105 |

0.2.7 decodes ~10% slower on this split (3090 + 5070 Ti); prompts are level or slightly faster. Not warm-up (run 2 matches), not the worker count (13 = 8), not drafting (acceptance unchanged: code 77-78%, agent 64-65%). Jim chose to stay on 0.2.7 (newer engine, DeepSeek support) and see whether it does better after the EPYC upgrade. To go back: `git checkout v0.1.43` in `~/lab/Strata` and re-run setup with `--build`. 0.2.7's setup adds no new options to the config; only the RAM-saving settings above differ from its own.

**After the EPYC upgrade (2026-10-30, 0.2.7):** EPYC 7443, 251 GB RAM, the 3090 and 5070 Ti at PCIe 4.0 x16 (Strata measures 26.5 GB/s, was 12.8).

| | old box, 0.2.7 (avg of 2) | EPYC, 62 GB config | EPYC, RAM limits removed (production) |
|---|---|---|---|
| 1K / 4K prompt | 437 / 1429 | 749 / 1967 | 774 / 2080 |
| 16K / 32K prompt | 2539 / 3164 | 3103 / 3681 | 3151 / 3711 |
| decode: code / prose / explain | 126 / 102 / 110 | 134 / 106 / 115 | 151 / 117 / 127 |
| agent: prompt / decode | 2400 / 103 | 3066 / 107 | 3094 / 120 |

Production now drops `--mmap-experts` and `--resident-budget-gib 30`, the 62 GB-era limits, so every expert the cards don't hold is pinned in RAM (~20 GB more). Pool workers 8, 16 and 22 measured equal (`sw-arena-w*`), so 8 stays. The prefill gain is the x16 links; the decode gain is the RAM.

Older results (fork era, 0.1.37 and 0.1.40, IQ3_XXS) are in `ab/results/` and `ab/ab040/`; their prompts came from the Strata source of the day, so only their decode figures compare directly.

## Updating Strata

```sh
cd ~/lab/Strata
git pull --ff-only
./setup.sh --family qwen --model IQ3_S --gguf-dir /mnt/data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S \
    --context 262144 --kv int8 --vision no --gpus 2,1 --layer-split auto --yes --no-start
```

Setup compiles the engine (`engine/strata`, for sm_86 and sm_120), updates the `.venv` and rewrites its own `strata-iq3_s.json`. Unload Strata first so its GPU checks see free cards. Compare setup's new config with the tuned ones for new options, then restart Strata and re-run the A/B.

## A/B testing

```sh
cd ~/llama-swap/strata/ab
STRATA_MODEL=Strata-IQ3S ./bench.py LABEL                  # the config llama-swap runs
STRATA_MODEL=Strata-IQ3S ./sweep.py ../strata-iq3_s.json base: pool13:"--pool-workers 13"
```

`sweep.py` writes each point's config to `ab/strata-sw-NAME.json`, points `ab.env` at it (read by `run-strata.sh`), restarts the model, benchmarks, and at the end removes `ab.env` and restarts on the production config. If a sweep is killed, delete `ab.env` by hand.
