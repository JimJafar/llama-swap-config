# llama-swap-config

llama-swap (llama.cpp) model configurations designed to run well on my 2x 16Gb Blackwell CUDA GPU machine.

## Hardware

| Slot | GPU | TDP | VRAM | Link (max) | Root |
|------|-----|-----|------|------------|------|
| `0000:02:00.0` | RTX 5070 Ti | 300 W | 16 GB | PCIe 5.0 x16 | CPU (Arrow Lake-HX Root Port `00:06.0`) |
| `0000:83:00.0` | RTX 5060 Ti | 180 W | 16 GB | PCIe 4.0 x8 | PCH (800-series Root Port `80:1d.0`) |

PSU: 1050 W.

The 5060 Ti is **chipset-attached** rather than on CPU-direct lanes; it shares DMI bandwidth with the rest of the PCH and is more sensitive to power and signal-integrity margin than the 5070 Ti.

## Resolved: second GPU falling off the bus under inference load

**Root cause (confirmed June 2026): the 5060 Ti was improperly seated.** Reseating it resolved the failure. Everything below was diagnosed *before* that was found; the `pcie_aspm=off` kernel parameter cut the failure rate by reducing PCIe link-power transitions, but it was masking a mechanical fault, not curing one.

> **Recurred under tensor parallelism (2026-06-09), now fixed (2026-06-10) — different cause.** Resolved for normal **layer-split** inference, but enabling `-sm tensor` dropped the 5060 Ti again, almost immediately. That was *not* the seating/ASPM fault (both already fixed) — it was the card's **chipset PCIe 4.0 x1 slot** being unable to take TP's sustained per-layer all-reduce traffic. Fixed physically by moving it to a CPU-direct Gen4 x4 link via an M.2 riser: see [TP stability](#tp-stability-5060-ti-x1-slot-drop-fixed-by-an-m2-riser).

### Symptom

While `qwen3.6-27B-Q6` was loaded (splits weights across both GPUs via `--tensor-split 1,1`), the 5060 Ti would disappear from `nvtop` mid-inference. Kernel log:

```
NVRM: Xid (PCI:0000:83:00): 79, GPU has fallen off the bus.
NVRM: GPU 0000:83:00.0: GPU has fallen off the bus.
```

Accompanied by sustained **Correctable PCIe Data Link Layer Timeout** errors on the same device, both at boot and during inference:

```
nvidia 0000:83:00.0: PCIe Bus Error: severity=Correctable, type=Data Link Layer
device [10de:2d04] error status/mask=00001000/0000e000
[12] Timeout
```

Graphical glitches were also visible during inference workloads.

### Diagnostic signature

When the card drops, `nvidia-smi` can no longer get a device handle and the PCIe link has **de-trained from x8 to x1**:

```
$ nvidia-smi
Unable to determine the device handle for GPU 0000:83:00.0: Unknown Error

$ cat /sys/bus/pci/devices/0000:83:00.0/current_link_width   # 1   (max_link_width = 8)
$ cat /sys/bus/pci/devices/0000:83:00.0/current_link_speed   # 16.0 GT/s — normal Gen4; the *width* is the tell
```

The link-**width** collapse (x8 → x1) under load — with healthy power draw (~150 W of 180 W) and cool temps — points at marginal physical contact rather than power or ASPM. That is what the reseat confirmed. The link speed staying at a correct Gen4 16 GT/s rules out a generation-negotiation problem.

**Recovery without a reboot** (sometimes re-trains the link; often a wedged card needs a reboot — and a `rescan` can hang if the device is truly unresponsive):

```
echo 1 | sudo tee /sys/bus/pci/devices/0000:83:00.0/remove
echo 1 | sudo tee /sys/bus/pci/rescan
```

### The actual fix

**Reseat the 5060 Ti.** A marginal seat holds at idle but de-trains under the combined link + power activity of inference, producing the Xid 79 / Data Link Layer timeouts above.

### Mitigation that masked it (kept — harmless): `pcie_aspm=off`

Added to the limine bootloader config:

```
pcie_aspm=off
```

- `pcie_aspm=off` — disables PCIe Active State Power Management, removing ASPM-driven L0s/L1 link transitions. This *reduced* the link drops by keeping the marginal link from cycling power states, which is why it originally looked like the cure — but it did not address the underlying mechanical fault. It is harmless and is left in place. No power-limit, BIOS PCIe-gen downgrade, or container privilege change is needed.

> **Note — `pcie_aspm=off` is system-wide.** It disables ASPM for *every* PCIe link in the machine (NVMe, NICs, etc.), not just the 5060 Ti. The cost is slightly higher idle power and heat on those devices; there is no performance penalty (latency is marginally better). If reclaiming ASPM elsewhere matters, scope it to `0000:83:00.0` via sysfs/`setpci` instead, or use `pcie_aspm.policy=performance`.

> **Discarded — `nvidia.NVreg_EnableGpuFirmware=0`.** This was originally also set, on the theory that disabling the GSP (GPU System Processor) firmware path cleared the graphical glitches. It does **not** do that here and has been removed. On Blackwell the NVIDIA **Open Kernel Module** is mandatory, and that module *requires* GSP — so the flag is silently ignored: `nvidia-smi -q` still reports an active `GSP Firmware Version` on both GPUs, and the parameter never even reached the live `/proc/cmdline`. The glitches were a downstream symptom of the link instability and cleared once the link was stable (ultimately, once the card was properly seated).

### Things that did *not* help

- **Forcing PCIe Gen 3** on the 5060 Ti slot in BIOS — made stability *worse*, not better. Left at Gen 4 (the slot's max). In hindsight, consistent with run-to-run noise over a marginal seat.
- **Power-limiting the 5060 Ti to 150 W** — reduced but did not eliminate the link errors. Consistent with the fault being mechanical (the seat), not power.

## ⚠️ Gotcha: `-ub` must not exceed `-b`

`-ub` (`--ubatch-size`, the *physical* micro-batch) must be **≤** `-b` (`--batch-size`, the *logical* batch). I originally had:

```
-b 2048
-ub 4096    # WRONG: micro-batch larger than batch
```

This is inconsistent — a micro-batch cannot be bigger than the batch it's drawn from. Worse, **the prefill compute buffer scales with `-ub`**, and that buffer is *not* split evenly across GPUs. On the 2x16 GB split this inflated the load on the 5060 Ti until large prompts hit:

```
CUDA error: out of memory
```

The OOM only showed up on **large prompts**, because that's when the oversized prefill buffer is actually allocated — small prompts fit fine and hid the bug.

### Rule of thumb

- Keep `-ub` ≤ `-b` (e.g. `-b 2048 -ub 512`). A smaller `-ub` shrinks the prefill compute buffer at a small prefill-speed cost — the main lever when you OOM only on long prompts.
- `--tensor-split 1,1` balances **weights** only. KV cache and the prefill compute buffer land unevenly, so the nominally-equal split can still saturate one card first. Nudge the split toward the underloaded GPU (e.g. `--tensor-split 1.1,1`) and watch `nvidia-smi` until both cards sit at similar used-MiB.
- The display/desktop also consumes ~750 MiB on GPU 0, further skewing the "equal" split.

### Tensor parallelism (`-sm tensor`) — WORKS, and is the fastest option

> **History:** this section previously said TP was "not usable" because the no-P2P
> `NODE` topology blocked NCCL. **That was wrong, twice over.** TP works on this
> machine, and `-sm tensor` + MTP is the *fastest* config measured for the 27B.
> The real blocker was a **stale NCCL in the llama.cpp image**, not the hardware
> and not the driver. Details below; the working entry is `Qwen3.6-27B-Q5-MTP-TP`.

**Measured decode throughput** (Qwen3.6-27B Q5_K_XL, 2×16 GB, identical prompt):

| split mode | no MTP | + MTP |
|---|---|---|
| `-sm layer` (pipelined, default) | 25.2 tok/s | ~35 tok/s |
| `-sm row` (TP, **deprecated**) | 20.3 tok/s | ~29 tok/s |
| **`-sm tensor` (TP)** | **~30 tok/s** | **~39.5 tok/s** ✅ |

> The `-sm tensor` row above was measured over the 5060 Ti's old **x1** link. After the
> M.2 riser put it on a CPU-direct **Gen4 x4** link (2026-06-10), `-sm tensor` + MTP
> reaches **54–64 tok/s** — see [TP stability](#tp-stability-5060-ti-x1-slot-drop-fixed-by-an-m2-riser).

So real tensor parallelism is **the fastest config** — already ~13% over layer+MTP on
the x1 link, and ~1.5–1.8× after the riser. (`-sm row` is the *old* row-split TP — it
works but is both deprecated and slower; ignore it.) `-sm tensor` also splits the KV
cache across both cards.

**What actually blocked it.** Three red herrings and one real cause:

1. **Topology / no-P2P is NOT the blocker.** The two GPUs are on separate PCIe host
   bridges (`nvidia-smi topo -m` shows `NODE`, no P2P), but NCCL connects fine over
   its **SHM transport** — the logs show `Channel 00 : 0[0] -> 1[1] via SHM/direct`
   and `Init COMPLETE`. This just needs `--ipc=host` to lift Docker's 64 MB
   `/dev/shm` cap (plus `NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0` to match the
   known-good config; these may be droppable on newer NCCL).
2. **The 610.x beta driver is NOT the blocker** (it was suspected — it's fine here).
3. **NCCL tuning is NOT the blocker** — the failure was invariant to `NCCL_PROTO`,
   `NCCL_ALGO`, `NCCL_CGA_CLUSTER_SIZE`, `NCCL_NVLS_ENABLE`, `NCCL_LAUNCH_MODE`, and
   `GGML_CUDA_DISABLE_GRAPHS`.
4. **The real cause: the image's bundled NCCL 2.25.1 has a broken `sm_120`
   (RTX 50-series) kernel launch.** NCCL connects, then aborts at the *first*
   all-reduce:

   ```
   ggml_backend_cuda_comm_allreduce_nccl ... ncclGroupEnd()
   enqueue.cc:1500 NCCL WARN Cuda failure 1 'invalid argument'
   ```

   (NCCL 2.25.1 *does* ship `sm_120` cubins, so it's not a missing-arch gap — the
   launch path itself is buggy for consumer Blackwell.) **Fix: preload a newer NCCL.**
   NCCL **2.30.4** makes `-sm tensor` work flawlessly.

**The fix, baked into `Qwen3.6-27B-Q5-MTP-TP`:**

```
--ipc=host                                              # NCCL SHM transport (no P2P needed)
-e LD_PRELOAD=/opt/newnccl/libnccl.so.2                 # use NCCL 2.30.4, not the image's 2.25.1
-v .../vendor/libnccl.so.2.30.4:/opt/newnccl/libnccl.so.2:ro
-e NCCL_P2P_DISABLE=1 -e NCCL_CUMEM_ENABLE=0
-sm tensor --fit off                                    # --fit is unimplemented for SPLIT_MODE_TENSOR
```

The 412 MB NCCL lib lives in `vendor/` (git-ignored). Refetch with:
`pip download --no-deps nvidia-nccl-cu12` then unzip the wheel
(`nvidia/nccl/lib/libnccl.so.2`).

**Fit caveat:** the 5070 Ti (`CUDA1` — see the GPU-order note below) carries the
desktop + the MTP draft context and fills first. If it OOMs, lower `-c` or shift weight
onto the 5060 by raising the *first* split number. See "Balancing & context" below —
and note the VRAM-fill `-c` is **not** the safe operating point.

**Tuning lever:** the cards are mismatched (5070 Ti vs 5060 Ti) and TP syncs every
layer, so the fast card waits on the slow one — a decode-speed cap no split fixes. The
split/`-c` tuning in "Balancing & context" below is about VRAM balance and stable
context, not speed.

### TP stability: 5060 Ti x1-slot drop, fixed by an M.2 riser

✅ **RESOLVED (2026-06-10).** TP is now stable *and* the fastest 27B config:
**54–64 tok/s** decode (prompt-dependent), no bus drops, both GPUs at **93–96%
utilisation** under load. What follows is the saga, kept as a record.

**The problem (2026-06-09).** With the 5060 Ti in its chipset **PCIe 4.0 x1** slot,
running `Qwen3.6-27B-Q5-MTP-TP` dropped it off the bus almost immediately — where
layer-split had run stable for a week. Different trigger from the
[earlier seating fault](#resolved-second-gpu-falling-off-the-bus-under-inference-load)
(`pcie_aspm=off` was already applied, so not ASPM):

```
$ nvidia-smi
Unable to determine the device handle for GPU1: 0000:84:00.0: Unknown Error
$ cat /sys/bus/pci/devices/0000:84:00.0/current_link_width   # 63  (garbage — link dead)
```

**Root cause: TP's traffic pattern killed the x1 link.** Layer split passes one
activation handoff per layer boundary (light). `-sm tensor` does an **all-reduce
every layer** (bidirectional, sustained) — the worst case for a narrow, marginal
link. A x1 link has no lanes to spare, so PCIe errors under continuous traffic
cascaded to an unrecoverable drop. It survived benchmarking (~39 tok/s) then dropped:
it failed *after* sustained stress, not instantly.

**The fix: 5060 Ti moved to a CPU-direct x4 link via an M.2 riser.** Put it on
**`M2A_CPU`** (CPU-connected PCIe 5.0 x4) through an **M.2 → PCIe x4 riser**;
relocated the slow NVMe that was there to a PCIe x1 slot via an x1→M.2 adapter. The
B860 DS3H WIFI6E manual confirmed `PCIEX16` and `M2A_CPU` are on **separate dedicated
CPU lanes** (no bifurcation note), so the 5070 Ti kept full x16.

**Measured outcome (better than predicted):**

```
$ cat /sys/bus/pci/devices/0000:01:00.0/current_link_width   # 4
$ cat /sys/bus/pci/devices/0000:01:00.0/current_link_speed   # 16.0 GT/s  (Gen4 — held under load)
$ nvidia-smi topo -m                                         # GPU0<->GPU1 = PHB (was NODE)
```

- Link trained at **Gen4 x4** (~7.9 GB/s), not the Gen3 we'd have accepted — and
  **holds at x4 during the all-reduce**. 5070 Ti stayed at Gen5 x16.
- Decode jumped to **54–64 tok/s** (from ~39 over x1, and ~35 layer+MTP). Bigger than
  the "x4 adds little to decode" prediction — because the real TP throttle wasn't
  bandwidth (the per-token all-reduce payload is tiny) but **per-layer sync latency +
  DMI contention**. Going CPU-direct (`NODE → PHB`) cut the round-trip; both GPUs now
  sit at 93–96% util instead of stalling on each sync.

**⚠️ GPU-order gotcha (bit us, twice).** After the riser the 5060 Ti enumerates on the
lower PCI bus (`01:00`), so `nvidia-smi` lists it as GPU0. We *assumed* llama.cpp's
**FASTEST_FIRST** order would still rank the bigger 5070 Ti as `CUDA0` — **it does not.**
Both cards are Blackwell (sm_120) and tie on compute capability, so FASTEST_FIRST falls
back to PCI-bus order: **`CUDA0` = 5060 Ti (`01:00`), `CUDA1` = 5070 Ti (`02:00`)** —
same as nvidia-smi. Confirmed empirically: biasing the split onto `CUDA1` OOMs the MTP
draft context on the 5070. So `--tensor-split a,b` = `5060,5070`; raise the **first**
number to shift load onto the 5060. The TP entry now pins
`-e CUDA_DEVICE_ORDER=PCI_BUS_ID` to keep this deterministic across reboots and driver
updates.

**Balancing & context (tuned 2026-06-11).** The 5070 Ti (`CUDA1`) is overhead-bound —
it carries the desktop + the MTP draft context, so it parks ~82–84% almost regardless
of the split, and pushing *weight* onto it (raising the 2nd number) OOMs the draft. The
5060 Ti (`CUDA0`) is the elastic card — tiny split changes swing it a lot. So the split
only *balances* the pair; the knob that adds total VRAM is `-c`, since KV cache lands on
both cards by the split ratio and fills the 5070 headroom the split can't reach.
Balancing to ~equal near `--tensor-split 1.01,1` then raising `-c` reached **89% (5070)
/ 96% (5060) at ~185k context** — but that's the *VRAM-fill* point and it **OOMs under a
real long-context prefill**. Backed off to a conservative **`-c 80000`** for headroom.
Lesson: tune the split for balance, but keep `-c` well below the fill point — the safe
ceiling is meaningfully lower than what loads at idle.

`Qwen3.6-27B-Q5-MTP-TP` is now a strong candidate to **become the default 27B** —
fastest and stable. Left as an explicit variant pending a longer no-drop soak.

## Deployment

llama-swap runs as a **host binary** from the `llama-swap-bin` AUR package (`/usr/bin/llama-swap`), driven by a systemd unit. It launches **each model as its own docker container** and stops it on swap (`cmdStop`), so only one model holds VRAM at a time:

- **llama.cpp models** use the `ghcr.io/mostlygeek/llama-swap:cuda` image with the entrypoint overridden to `/app/llama-server` (the image's entrypoint is `llama-swap` itself).
- **vLLM models** (NVFP4 + MTP) are **parked** — not currently wired up. See "Parked: NVFP4 + MTP via vLLM" below for why and how to resume.

The systemd unit is tracked at [`llama-swap.service`](llama-swap.service) and installed to `/etc/systemd/system/llama-swap.service` (it shadows the AUR vendor unit at `/usr/lib/systemd/system/`, which uses `DynamicUser=yes` and so can't access docker). It runs as `jim` (in the `docker` group), listens on `0.0.0.0:8033`, and uses `-watch-config` so `config.yaml` edits live-reload without a restart.

## Updating

Three independently-versioned pieces:

```
paru -S llama-swap-bin                 # 1. llama-swap host binary (AUR)
./update.sh                            # 2. llama.cpp image (docker pull mostlygeek:cuda)
./version.sh                           # report installed versions
# (vLLM image only needed if/when the parked NVFP4 path is revived)
```

`version.sh` reports the host llama-swap version and the llama.cpp build inside the image (the web UI shows neither). Example output:

```
llama.cpp:  version: 9468 (354ebac8c)
llama-swap: version: 223 (...)
```

## Parked: NVFP4 + MTP via vLLM

Goal: run `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` (modelopt NVFP4 + multi-token-prediction) to get native Blackwell FP4 + MTP speculative decoding. **The model does not initialize in vLLM 0.22.0 on this machine — but the reason is the *model*, not the hardware.**

> ### ⚠️ Correction (2026-06-08) — the original "no-P2P blocks TP" conclusion was WRONG
>
> The earlier diagnosis below blamed the no-P2P `NODE` topology. A controlled isolation pass disproved that:
>
> - **Bare 2-GPU NCCL works.** A 2-rank `torch.distributed.all_reduce` in the same `vllm/vllm-openai:latest` image completes correctly across both GPUs **`via SHM/direct/direct`** (NCCL's shared-host-memory transport), with `NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 --ipc=host`. Init 0.2 s, correct result. **No P2P is needed — NCCL falls back to SHM.** So multi-GPU vLLM/TP is *not* blocked here.
> - **The model hangs identically under TP=2, PP=2, *and a single GPU*** (`CUDA_VISIBLE_DEVICES=0 -tp 1 --cpu-offload-gb 5`). Same freeze point every time: right after kernel selection (`FlashInferCutlassNvFp4LinearKernel` for NVFP4 + `Triton/FLA GDN` for the gated-delta-net linear attention), before `Model loading took` ever prints. CPU sits ~idle (a real deadlock, not slow JIT).
> - **Conclusion:** since it deadlocks on a single GPU with no NCCL at all, the blocker is the **GDN (gated-delta-net) linear-attention init in this vLLM build**, not the interconnect. The "TP deadlocks at the first all-reduce" finding was a misattribution — `NCCL_DEBUG=INFO` printed no transport lines because vLLM never *reached* the collective; it hung earlier, the instant after `qwen_gdn_linear_attn.py:228 Using Triton/FLA GDN prefill kernel`.
> - **It is quant-independent and has no config workaround.** Reproduced identically with NVFP4 *and* GPTQ-Int4/Marlin, under TP=2 / PP=2 / single-GPU, with MTP on/off, `--enforce-eager`, `--max-num-batched-tokens 2096`, and `--gpu-memory-utilization` from 0.45–0.92 (the "GDN Triton-autotuner needs free VRAM" theory was tested with ~9 GB free and 12 GB offloaded — **GPU util never spiked**, so the autotuner never even runs; the hang precedes it).
> - **Not a version regression, and not a missing kernel.** Tested vLLM **v0.20.0** (the exact version a working 2×5060 Ti report used — [note.com 30 tok/s](https://note.com/cute_agapan9087/n/nb4b3456ca8b4)) **and v0.22.0** — both hang at the same GDN line. Replicating that report's command verbatim (`--trust-remote-code`, torch.compile path, `qwen3_5_mtp`, gpu-mem-util 0.88) didn't help.
> - **Narrowed to the FLA (flash-linear-attention) GDN kernel specifically.** On this host: bare NCCL all-reduce works ✅, and a minimal `@triton.jit` kernel compiles+runs in 0.5 s ✅ (`triton 3.6.0 / torch 2.11.0+cu130`). So CUDA, the driver, NCCL, and *basic* Triton are all fine — only the FLA GDN linear-attention kernel deadlocks at first use. Quant-, parallelism-, and config-independent (NVFP4 + GPTQ-Int4/Marlin; TP/PP/single-GPU; MTP on/off; eager/compile; gpu-mem-util 0.45–0.92).
> - **Most likely cause: this host's environment, not vLLM.** Since the same model+version runs for others on sm_120, the differences here are the prime suspects: **bleeding-edge driver `610.43.02`** and the **CachyOS custom kernel** (vs the working report's WSL2 + older driver). The FLA autotune/compile path appears to hang on this driver/kernel combo.
>
> **Implications:** (1) TP across these two GPUs is viable (slow, via SHM) — `--disable-custom-all-reduce` + `--ipc=host`. (2) The blocker is **this host's FLA-GDN environment, not the quant, the hardware, or the vLLM version.** Highest-probability fix: **a stable production NVIDIA driver** (610.x is beta-grade) and/or a **stock/LTS kernel**, then retry the note.com command. Secondary: a community sm_120 vLLM fork (`aliez-ren/vllm-qwen3.5-nvfp4-sm120`) or alt engines (`kekzl/imp`, `devnen/qwen3.6-windows-server`). Until then, **llama.cpp** is the working path (it implements GDN itself, no Triton/FLA dependency). See the corrected `configure-vllm` skill for the general no-P2P/TP rule.

Original findings (June 2026, vLLM 0.22.0 — kept for history, see correction above):

- ~~**Single-GPU: impossible** (OOM, needs ≥24 GB).~~ It OOMs on weights *only if you don't offload*; with `--cpu-offload-gb` it loads and then hangs at model init — same as multi-GPU.
- **Dual-GPU pipeline-parallel: rejected** — MTP's draft model doesn't implement vLLM's `SupportsPP` (`NotImplementedError`). *(Still true — but the base model also hangs under PP for the init reason above, so PP+MTP was never the real wall.)*
- ~~**Dual-GPU tensor-parallel: deadlocks on the no-P2P `NODE` topology.**~~ **Reattributed:** the deadlock is the model's GDN linear-attention init (quant-independent), reproducible on a single GPU. NCCL/TP itself works fine here (bare all-reduce proven).

The board having one CPU-wired slot (`NODE`, no P2P) is real, but it does **not** prevent vLLM TP — NCCL uses its SHM transport. A second CPU-wired slot or a ≥24 GB GPU would help *performance/fit*, not *feasibility*. Meanwhile the working dual-GPU path remains **llama.cpp layer-split** (`Qwen3.6-27B-Q5-MTP`).

Best-known command to resume from *if the hardware ever changes* (furthest-progressing config; still hangs on the current board for the reason above):

```
docker run --rm --runtime nvidia --ipc=host \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=0,1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e NCCL_P2P_DISABLE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
  -p 8000:8000 -v /home/jim/models/vllm:/root/.cache/huggingface \
  <vllm-image> \
  --model sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP --served-model-name qwen36-nvfp4-mtp \
  --tensor-parallel-size 2 --max-model-len 204800 --max-num-batched-tokens 8192 --max-num-seqs 1 \
  --gpu-memory-utilization 0.88 --kv-cache-dtype fp8 --quantization modelopt \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 --language-model-only --generation-config vllm \
  --disable-custom-all-reduce --attention-backend TRITON_ATTN --port 8000
```

Once it serves on `:8000`, wiring it into llama-swap is just a `cmd:`/`cmdStop:` model entry wrapping the above (publish `${PORT}:8000`).

## Upgrade options & expected gains

Analysis of what would actually move the needle, given the current bottleneck is the **5060 Ti** (448 GB/s, ~half the 5070 Ti's bandwidth) and the **single CPU-wired slot**.

> **Note (2026-06-08):** the framing below treats "no GPU P2P" as a hard *feasibility* blocker for vLLM TP. Per the correction in "Parked: NVFP4 + MTP via vLLM", that's wrong — vLLM TP already *works* here over NCCL's SHM transport. So a second CPU-wired slot / P2P would improve **performance** (lower all-reduce latency), not unlock a capability that's currently impossible. The bandwidth ceiling and the mismatched-card sync penalty below are still the real limiters.

Baseline today: **`Qwen3.6-27B-Q5-MTP`, llama.cpp layer-split, ~36–47 tok/s, 156k context.**

### M.2 → PCIe x4 riser for the 5060 Ti (cheapest — enabled stable TP) ⭐ DONE
- **Done (2026-06-10):** moved the 5060 Ti off its chipset **PCIe 4.0 x1** slot onto **`M2A_CPU`** (CPU-direct) via an **M.2 → PCIe x4 riser**; relocated the slow NVMe to a PCIe x1 slot via an x1→M.2 adapter. Full detail in the TP section's [stability writeup](#tp-stability-5060-ti-x1-slot-drop-fixed-by-an-m2-riser).
- **Delivered:** the 5060 Ti's link went **x1 → Gen4 x4** (~7.9 GB/s) and onto CPU lanes (`NODE → PHB`), and **holds under load** — `-sm tensor` no longer drops the card. Also ended the chipset/DMI bus-fragility.
- **Measured decode:** **`-sm tensor` + MTP = 54–64 tok/s** (vs ~35 layer+MTP) — ~1.5–1.8×, bigger than predicted because the x1 link's *latency*, not bandwidth, was the real TP throttle.
- **Notes:** riser cable quality was the stability variable (a good shielded one held full Gen4); needs a powered adapter; the 5070 Ti kept full x16 (manual-confirmed, no bifurcation). Did **not** help vLLM (that's blocked by the FLA-GDN issue, not the interconnect).

### Z890 board with two CPU-wired x8/x8 slots
- **Unlocks:** both GPUs on CPU lanes → P2P → vLLM tensor-parallel works → **NVFP4 + MTP across both cards** becomes possible. Also removes the 5060 Ti from the chipset/DMI link (ends the "fallen off the bus" fragility). *(Note: the M.2-riser option above already gets the 5060 Ti onto CPU lanes for far less — the Z890's extra value is the full x8 width and a clean second slot, not a unique capability.)*
- **Realistic decode:** **~50–70 tok/s (≈1.3–1.7×)**, *not* 2×. Gains come mostly from NVFP4 (~20–30% less bandwidth/token) and native FP4 **prefill** (the long-prompt win); TP parallelism adds less than hoped.
- **Why it's capped:**
  - **No NVLink** — P2P would be over **PCIe Gen4 x8 (~16 GB/s, capped by the Gen4 5060 Ti)**. TP all-reduces every layer; that interconnect latency eats much of the parallelism. Consumer no-NVLink TP often nets only ~1.2–1.5× over layer-split for single-stream.
  - **Mismatched cards** — TP syncs every step, so the 5070 Ti waits on the 5060 Ti. No board fixes this. (This is why matched 2×3090 setups hit ~70 tok/s and this pair won't.)

### Ranked by speed-per-spend
1. **M.2 → PCIe x4 riser (5060 Ti onto `M2A_CPU`)** — ✅ done, by far the cheapest. Made llama.cpp `-sm tensor` TP *stable* (**54–64 tok/s** vs ~35 layer+MTP) and ended bus-fragility, with zero new silicon. Everything below is a bigger spend for less marginal gain.
2. **Single ≥24 GB GPU** (e.g. used 3090/4090-class or a 24 GB Blackwell) — biggest gain: runs the model on one fast card, no inter-GPU penalty at all, and unlocks single-GPU NVFP4+MTP. Removes both bottlenecks.
3. **Replace the 5060 Ti with a matched 5070 Ti** — fixes the bandwidth ceiling *and* TP balance; would need the Z890 board too for P2P.
4. **Z890 x8/x8 board (5060 Ti kept)** — unlocks the most vLLM *capability* (NVFP4+MTP) but delivers little *speed* (~1.3–1.7×), and the riser already gets the 5060 Ti onto CPU lanes for a fraction of the cost. Good only if the goal is the NVFP4/vLLM path + full x8 width.

Bottom line: **the riser was the win** — cheapest unlock, fixed the TP stability problem, and `-sm tensor`+MTP now runs 54–64 tok/s. Beyond that, the 5060 Ti remains the ceiling, so for more raw speed upgrade the *GPU* before the board.
