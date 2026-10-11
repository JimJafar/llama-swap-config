#!/usr/bin/env bash
# RAM guard for Strata.
#
# WHY: on 2026-09-30 a Strata run drove MemAvailable to ~1 GiB and the kernel's
# OOM killer chose Paseo (a desktop app) as the victim. Strata's expert arena is
# ~43 GiB of anonymous memory with the default 8 MiB memlock limit, so it is
# swappable and the box can thrash. This guard makes Strata the victim instead
# of your desktop: if MemAvailable falls below a threshold, it unloads Strata
# and leaves everything else alone.
#
# Usage:  setsid nohup /home/jim/llama-swap/strata/ram-guard.sh [threshold_MB] &
# Stop:   pkill -f '[r]am-guard.sh'
# It does not survive a reboot, and it only acts when Strata is actually loaded.
set -u
THRESHOLD_MB="${1:-2048}"
MODEL=Strata-IQ3S
ENDPOINT="http://127.0.0.1:8033/api/models/unload/${MODEL}"

log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }

log "guard started (threshold ${THRESHOLD_MB} MB)"
while true; do
    avail_kb=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
    avail_mb=$((avail_kb / 1024))
    if [ "$avail_mb" -lt "$THRESHOLD_MB" ]; then
        # only act if Strata is genuinely loaded (bracket trick: do not match self)
        if pgrep -f '[e]ngine/strata' >/dev/null 2>&1; then
            log "MemAvailable ${avail_mb} MB < ${THRESHOLD_MB} MB -> unloading ${MODEL}"
            python3 - "$ENDPOINT" <<'PY'
import sys, urllib.request
url = sys.argv[1]
try:
    r = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(r, timeout=30) as x:
        print(f"  unload -> HTTP {x.status}")
except Exception as e:
    print(f"  unload failed: {type(e).__name__}: {e}")
PY
            sleep 60   # give it time to actually release before judging again
        fi
    fi
    sleep 10
done
