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

Two kernel parameters added to the limine bootloader config:

```
pcie_aspm=off nvidia.NVreg_EnableGpuFirmware=0
```

- `pcie_aspm=off` — disables PCIe Active State Power Management. ASPM-driven L0s/L1 link transitions were the source of the Data Link Layer timeouts on the chipset-attached slot.
- `nvidia.NVreg_EnableGpuFirmware=0` — disables the GSP (GPU System Processor) firmware path in the NVIDIA driver, falling back to the legacy kernel-driven init. Cleared the residual graphical glitches.

Together these eliminated the link drops, the Xid 79 failures, and the inference-time visual artifacts. No power-limit, BIOS PCIe-gen downgrade, or container privilege change is needed.

### Things that did *not* help

- **Forcing PCIe Gen 3** on the 5060 Ti slot in BIOS — made stability *worse*, not better. Left at Gen 4 (the slot's max).
- **Power-limiting the 5060 Ti to 150 W** — reduced but did not eliminate the link errors; the underlying ASPM/firmware issue was unaffected.
