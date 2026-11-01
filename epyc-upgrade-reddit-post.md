**Moved my local-LLM box from Arrow Lake to a used EPYC 7443: what I had to fix, and before/after numbers**

Same GPUs before and after: RTX 3090 FE, RTX 5070 Ti, 2× RTX 5060 Ti 16 GB (one is out waiting for a riser, so the big MoE runs below use 3 cards, against 4 before).

| | Before | After |
|---|---|---|
| CPU / board | Core Ultra 7 265KF, MSI Z890 | EPYC 7443 (24c, Zen 3, 4 CCDs), Supermicro H12SSL-NT |
| RAM | 64 GB DDR5-6000, 2 channels | 256 GB DDR4 ECC RDIMM, 8 channels (2666 sticks, run at 3200) |
| GPU links | x8 / x8 / x4 / x4 (chipset) | x16 / x16 / x8 |
| PSU | 1050 W | NZXT C1500 |

## What I had to fix or tune

1. **Fans surging every few seconds.** The BMC's default "lower critical" fan threshold (420 RPM) was above what quiet fans idle at, so it kept declaring them dead and ramping everything to 100%. Fix: `ipmitool sensor thresh FANx lower 100 200 300` for each connected fan, then `ipmitool mc reset cold`. The BMC rounds to 140 RPM steps.
2. **GPU power limits silently reset.** LACT stores limits by PCI address, and every card's address changed on the new board. Now a small `nvidia-smi` script sets them by GPU UUID from a systemd service and timer (3090 280 W, 5070 Ti 260 W, 5060 Tis 150 W). A sweep showed 280–300 W is the 3090's sweet spot for inference.
3. **GPU numbering changed** (twice). Everything that names cards now uses UUIDs and maps them to indices at launch.
4. **Drop the RAM-starved settings.** Engine options that existed to fit 64 GB (mmap'd experts, resident budgets) were holding back decode once there was 256 GB.
5. **RAM at 3200 instead of the rated 2666.** Triad 88.6 → 102 GB/s; GLM decode only about +3–5%. Stability: memtester (as root: as a normal user the memlock limit quietly shrinks it to ~8 MB per copy) plus a 3-hour DeepSeek run, with ECC at 0 corrected / 0 uncorrected throughout.
6. **NPS1, not NPS4.** llama.cpp-style engines aren't NUMA-aware. The STREAM gains people quote for NPS4 come from pinned per-quadrant workers.
7. **`iommu=pt` on the kernel command line.** +3.5% prefill for tensor-parallel llama.cpp, and NVMe random reads recovered (KIOXIA QD32 837K → 941K IOPS).
8. **BIOS:**
   - APBDIS = 1 with fixed SOC P-state P0, which keeps the Infinity Fabric at full clock (about +20 W at idle).
   - Determinism slider set to Power.
   - **Resizable BAR on** (it was off: BAR1 was 256 MiB).
   - PCIe Ten Bit Tag on.

   Together: another +4% tensor-parallel prefill and a few % on GLM decode.
9. **llama.cpp `-sm tensor`: `-ub 1024` instead of 512.** Without P2P, the two cards relay every layer's results through system RAM, across the I/O die. A bigger micro-batch halves the number of relays: +5% on Qwen3.8-27B, +9% on a 27B fine-tune. It needed a little more VRAM, so the split moved slightly toward the 3090.

**Didn't help, or made things worse:**
- **Performance governor:** no change.
- **`NCCL_SHM_USE_CUDA_MEMCPY=1`:** it **hangs** llama.cpp tensor-parallel (first request never finishes).
- **More CPU threads for GLM's CPU expert lane:** flat above ~16 threads, so I'm not upgrading to an 8-CCD chip for this.
- **Disabling ACS:** pointless without PCIe switches or P2P.

## Before / after (latest numbers, after all of the above)

| Test | Before | After | Change |
|---|---:|---:|---:|
| RAM bandwidth, triad, all threads | 58.8 GB/s | 102.0 GB/s | **+73%** |
| RAM bandwidth, 1 thread | 32.3 GB/s | 37.4 GB/s | +16% |
| PCIe host↔GPU, 3090 / 5070 Ti | 12.8 / 13.6 GB/s | 26.8 / 28.4 GB/s | **~2×** |
| PCIe, 5060 Ti | 6.4 GB/s | 14.1 GB/s | **2.2×** |
| **GLM-5.3-Flash** (Project Maya, IQ3_S): decode | 9.7 tok/s | 41.0 tok/s | **4.2×** |
| GLM-5.3-Flash: 29K prompt prefill / decode after it | 577 / 8.2 | 1,069 / 37.8 | **1.9× / 4.6×** |
| **DeepSeek-V4.1-Flash** (Project Maya, Maya-S): decode | 9.3 tok/s | 35.2 tok/s | **3.8×** |
| DeepSeek-V4.1-Flash: 29K prompt prefill / decode after it | 320 / 6.8 | 1,064 / 31.2 | **3.3× / 4.6×** |
| **Strata** (Qwen3.8-Flash-Next IQ3_S): 1K / 32K prompt prefill | 437 / 3,164 tok/s | 808 / 3,756 tok/s | +85% / +19% |
| Strata: decode, code / agent (20K context) | 126 / 103 tok/s | 149 / 119 tok/s | +18% / +16% |
| **Qwen3.8-27B Q6_K**, llama.cpp tensor parallel (3090 + 5070 Ti): 64K prefill | 1,146 tok/s | 1,120 tok/s | −2% (was −13% before tuning) |
| Qwen3.8-27B: decode, short / 64K prompt | 72.6 / 76 tok/s | 70.5 / 78 tok/s | about the same |
| SSD random 4K read QD32, WD SN7100 (btrfs) | 102K IOPS | 381K IOPS | **3.7×** |
| SSD random 4K read QD32, KIOXIA (xfs) | 986K IOPS | 941K IOPS | −5% |
| SSD sequential read, both drives | ~7.1 GB/s | ~7.2–7.4 GB/s | same |

**Notes:**
- **Engine versions:** Maya and Strata were updated along the way. Strata's "before" is the same 0.2.x line; Maya went from 1.0.34 / 1.1.20 to 1.1.42.
- **Benchmark settings:** the Strata and 27B runs use temperature 0 with thinking off, so they're best-case decode figures. The comparisons are like for like.
- **The big MoE models** were bottlenecked on RAM capacity. Before, 30–60% of the model was read from the NVMe drive every token; now it all sits in VRAM plus RAM, which is where the 4× comes from.
- **Dense models on the GPUs gain nothing.** The 27B is limited by the card-to-card relay. It only got back to parity after the IOMMU, BIOS and `-ub` changes.
