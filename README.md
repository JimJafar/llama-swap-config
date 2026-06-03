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

### Do not use tensor parallelism (`-sm tensor`)

Tensor parallelism is **not usable on this machine** — use the default layer split (`-sm layer`). The two GPUs sit on **separate PCIe host bridges** (the 5070 Ti on CPU-direct lanes, the 5060 Ti on the PCH), so there is **no peer-to-peer path** between them:

```
$ nvidia-smi topo -m
       GPU0   GPU1
GPU0    X     NODE      # NODE = traverses PCIe host bridges (via the CPU/DMI) — no P2P
GPU1   NODE    X
```

`-sm tensor` runs on NCCL, which requires GPU↔GPU P2P. Without it, NCCL cannot establish its transport and `llama-server` aborts during warm-up:

```
CUDA error: unhandled cuda error (run with NCCL_DEBUG=INFO for details)
```

(Docker also caps the container's `/dev/shm` at 64 MB by default, which independently breaks NCCL's shared-memory fallback.) This is a hardware-topology limit, not a config bug — tensor parallelism needs matched GPUs with a P2P path (e.g. both on CPU-direct lanes, or NVLink). Stick to layer split here.

## Updating

The `ghcr.io/mostlygeek/llama-swap:cuda` image bundles **both** llama-swap and a pinned snapshot of llama.cpp's `llama-server`, so the llama.cpp version advances only when the image is re-pulled.

```
./update.sh    # docker pull ghcr.io/mostlygeek/llama-swap:cuda
./version.sh    # print the bundled llama.cpp + llama-swap versions
```

`version.sh` reads the versions straight from the running container (the llama-swap web UI shows neither). It locates the container by image, so it survives the container-name change you get after each re-pull. Example output:

```
llama.cpp:  version: 9468 (354ebac8c)
llama-swap: version: 222 (...)
```
