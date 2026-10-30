#!/bin/sh
# Set Marvin's GPU power limits, by GPU UUID (stable across slots and motherboards,
# unlike PCI addresses: the EPYC upgrade on 2026-10-30 moved every card and silently
# reset LACT's address-keyed limits to stock).
#
# Run as root by gpu-power-limits.service at boot and by gpu-power-limits.timer every
# 10 minutes (a GPU reset or driver reload drops the limits; re-applying is harmless).
#
# Why these values (Strata + 27B Q6 sweeps, 2026-10-29, strata/ab/powertest.py):
#   3090    280 W  sweet spot: 260-300 W is within ~3% on the 27B; below 240 W the clock
#                  collapses (200 W cost the 27B 35-40% decode); above 300 W ~1% per 15 W.
#   5070 Ti 260 W  never reached under Strata or the 27B (peak ~190 W), so this is a ceiling.
#   5060 Ti 150 W  each; the cards' minimum.
# Budget on the NZXT C1500: ~840 W of GPU limits + ~340 W for the rest = ~79% of 1,500 W.
set -u
apply() {   # UUID WATTS NAME
    if ! nvidia-smi -L | grep -q "$1"; then
        echo "$3 ($1): not installed, skipped"   # a card out of the machine is not an error
        return
    fi
    if nvidia-smi -i "$1" -pl "$2" >/dev/null 2>&1; then
        echo "$3 ($1): $2 W"
    else
        echo "$3 ($1): FAILED to set $2 W" >&2
        rc=1
    fi
}
rc=0
nvidia-smi -pm 1 >/dev/null 2>&1   # persistence mode: keep the driver's settings while GPUs idle
apply GPU-93ab52e5-8e61-197e-880a-aa6af5a13c74 280 "RTX 3090 FE"
apply GPU-d64df8cf-c877-b43d-d9a5-d6b0e38e504a 260 "RTX 5070 Ti (Palit)"
apply GPU-57bbb375-a61a-6662-e9c8-447b948f0833 150 "RTX 5060 Ti (MSI)"
apply GPU-af5bd53f-a1ca-d807-1ae2-7f5ef261fe1a 150 "RTX 5060 Ti (ZOTAC)"
exit $rc
