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

### Fix

The single effective kernel parameter, added to the limine bootloader config:

```
pcie_aspm=off
```

- `pcie_aspm=off` — disables PCIe Active State Power Management. ASPM-driven L0s/L1 link transitions were the source of the Data Link Layer timeouts on the chipset-attached slot. This eliminated the link drops, the Xid 79 failures, **and** the inference-time visual artifacts (the artifacts were a downstream symptom of the same link instability).

No power-limit, BIOS PCIe-gen downgrade, or container privilege change is needed.

> **Note — `pcie_aspm=off` is system-wide.** It disables ASPM for *every* PCIe link in the machine (NVMe, NICs, etc.), not just the 5060 Ti. The cost is slightly higher idle power and heat on those devices; there is no performance penalty (latency is marginally better). If reclaiming ASPM elsewhere matters, scope it to `0000:83:00.0` via sysfs/`setpci` instead, or use `pcie_aspm.policy=performance`.

> **Discarded — `nvidia.NVreg_EnableGpuFirmware=0`.** This was originally also set, on the theory that disabling the GSP (GPU System Processor) firmware path cleared the graphical glitches. It does **not** do that here and has been removed. On Blackwell the NVIDIA **Open Kernel Module** is mandatory, and that module *requires* GSP — so the flag is silently ignored: `nvidia-smi -q` still reports an active `GSP Firmware Version` on both GPUs, and the parameter never even reached the live `/proc/cmdline`. The glitches were fixed by `pcie_aspm=off` alone.

### Things that did *not* help

- **Forcing PCIe Gen 3** on the 5060 Ti slot in BIOS — made stability *worse*, not better. Left at Gen 4 (the slot's max).
- **Power-limiting the 5060 Ti to 150 W** — reduced but did not eliminate the link errors; the underlying ASPM issue was unaffected.

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
