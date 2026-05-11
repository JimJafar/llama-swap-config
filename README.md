# llama-swap-config

llama-swap (llama.cpp) model configurations designed to run well on my 2x 16Gb Blackwell CUDA GPU machine.

## Hardware

| Slot | GPU | TDP | VRAM | Link (max) | Root |
|------|-----|-----|------|------------|------|
| `0000:02:00.0` | RTX 5070 Ti | 300 W | 16 GB | PCIe 5.0 x16 | CPU (Arrow Lake-HX Root Port `00:06.0`) |
| `0000:83:00.0` | RTX 5060 Ti | 180 W | 16 GB | PCIe 4.0 x8 | PCH (800-series Root Port `80:1d.0`) |

PSU: 1050 W.

The 5060 Ti is **chipset-attached** rather than on CPU-direct lanes; it shares DMI bandwidth with the rest of the PCH and is more sensitive to power and signal-integrity margin than the 5070 Ti.

## Known issue: second GPU falls off the bus under qwen3.6-27B-Q6

### Symptom

While `qwen3.6-27B-Q6` is loaded (which splits weights across both GPUs via `--tensor-split 1,1`), the second GPU disappears from `nvtop` or stops returning telemetry. Only a host reboot recovers it.

### Evidence

1. **Xid 79** on the 5060 Ti — the kernel-driver signal for an unrecoverable GPU loss:

   ```
   NVRM: Xid (PCI:0000:83:00): 79, GPU has fallen off the bus.
   NVRM: GPU 0000:83:00.0: GPU has fallen off the bus.
   ```

2. **Correctable PCIe Data Link Layer Timeout errors** on the same device at boot (30+ entries):

   ```
   nvidia 0000:83:00.0: PCIe Bus Error: severity=Correctable, type=Data Link Layer
   device [10de:2d04] error status/mask=00001000/0000e000
   [12] Timeout
   ```

   On their own these are recoverable, but they indicate the link to this card is marginal.

3. The 5070 Ti shows no errors — only the chipset-attached card is affected.

### Likely causes (in order)

1. **Transient power sag.** Xid 79 with otherwise-healthy silicon usually traces to a power-delivery issue. Even with a 1050 W PSU, Blackwell cards have large transient spikes (often ~2× rated TDP for sub-millisecond windows) and `--tensor-split 1,1` keeps both cards near peak power *simultaneously*.
2. **Marginal PCIe link** on the PCH slot — see the correctable errors above.
3. NVIDIA open kernel driver `595.71.05` on new Blackwell silicon — possible but secondary.

### Mitigations to try (cheapest first)

- **Power-limit the 5060 Ti** to 150 W (its vBIOS minimum; default 180 W) before `llama-server` starts. Wired into `config.yaml` as a `sh -c` wrapper that runs `nvidia-smi -i 1 -pm 1 && nvidia-smi -i 1 -pl 150` before `exec llama-server`. If the GPU stops dropping, power was the trigger and the limit can be raised gradually toward 180 W.
- **Force PCIe Gen 4 (or Gen 3)** for slot 83:00.0 in BIOS to add link margin.
- **Reseat the card** and verify each 8-pin PCIe power lead is on its own PSU rail rather than a daisy chain.
- Monitor actual draw under load with `nvidia-smi dmon -s pucvmet`.

### Container permissions required for the hook

Setting power limits via NVML requires elevated capabilities. Tested combinations:

| Flag | Works? |
|------|--------|
| baseline (`--runtime nvidia`) | ❌ Insufficient Permissions |
| `-e NVIDIA_DRIVER_CAPABILITIES=all` | ❌ Insufficient Permissions |
| `--cap-add SYS_ADMIN` | ✅ |
| `--privileged` | ✅ (overkill) |

Running `docker run` itself under `sudo` does **not** help — the Docker daemon is already root, and what matters is the container's capability set, not the CLI invoker.

A systemd drop-in that adds `--cap-add SYS_ADMIN` to the `llama-swap.service` unit is staged at `systemd/override.conf`. Install with:

```bash
sudo mkdir -p /etc/systemd/system/llama-swap.service.d
sudo cp systemd/override.conf /etc/systemd/system/llama-swap.service.d/override.conf
sudo systemctl daemon-reload
sudo systemctl restart llama-swap.service
```

### Workarounds if the GPU drops again

- Weight the split toward the 5070 Ti, e.g. `--tensor-split 30,9`, until the root cause is fixed.
- Fall back to a single-card model (`qwen3.6-35B-Q4` uses `--fit on` and the 5070 Ti has enough VRAM to host smaller models alone).

### Progress after limiting to 150W

I am still seeing lots of this error:

```
  [ 4899.194591] nvidia 0000:83:00.0: PCIe Bus Error: severity=Correctable, type=Data Link Layer, (Transmitter ID)                                                                                                                                          
  [ 4899.194592] nvidia 0000:83:00.0:   device [10de:2d04] error status/mask=00001000/0000e000                                                                                                                                                              
  [ 4899.194592] nvidia 0000:83:00.0:    [12] Timeout                                                                                                                                                                                                       
```

That's the link recovering itself in real time — the same Data Link Layer timeouts we saw at boot, now firing under inference load. The link layer auto-retransmits, so they're not fatal on their own, but a marginal link is what eventually escalates into Xid 79.

Implication: power-limiting alone won't fix this. It may be enough to keep the card from falling off the bus (no Xid 79 = win), but you're running on a degraded link. Worth checking:

  - Does the model now run to completion without Xid 79? Even with these correctable errors, if no fallen-off-the-bus, you've gained stability.
  - The correctable error rate is hurting throughput too — every timeout triggers a PCIe link replay. Worth comparing tok/s against a single-card run.

Next-cheapest fix that targets the actual root cause: drop the slot's PCIe gen in BIOS. Look for something like "PCIe Slot Configuration → Slot for 5060 Ti → Link Speed: Gen5/Auto → Gen4" (or Gen3 if Gen4 still misbehaves). Slower theoretical bandwidth, but the slot is already only Gen4 x8 max, and a stable Gen3 link beats an unstable Gen4 one every time.

Two other things worth ruling out at the hardware level while you're in there:

  1. Reseat the card and the PCIe power cables. If either 8-pin lead is daisy-chained off one PSU output, split them to separate rails.
  2. Check if there's a riser involved. PCIe risers are a very common source of exactly these errors, especially at Gen4/Gen5 speeds.
  
  Keep dmesg -w | grep Xid running — if Xid 79 stays away through a full long-context run, the power limit alone is doing the job and you can defer the BIOS change.

