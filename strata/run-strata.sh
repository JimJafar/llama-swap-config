#!/bin/sh
# Strata launcher for llama-swap:  run-strata.sh PORT CONFIG [OTHER]
#
# Runs the official Strata checkout (~/lab/Strata: its serve/server.py, .venv and
# engine) with an engine config here (strata-iq3_s.json).
#
# OTHER (optional): another persistent llama-swap model that needs the same cards.
# llama-swap never evicts one persistent model for another, so this unloads OTHER
# itself, but only once it is idle (no request running or queued), up to
# STRATA_WAIT_S: llama-swap's unload API stops a model at once, which killed requests
# mid-answer. Unused since Strata-IQ3XXS was deleted (2026-10-12).
#
# A/B testing: ab.env next to this script, when present, is read after the arguments
# and may set CONFIG (another engine config) and engine env vars (exported). The
# harness in ab/ writes and removes it. Don't leave one behind.
set -eu
PORT="${1:?usage: run-strata.sh PORT CONFIG [OTHER]}"
CONFIG="${2:?usage: run-strata.sh PORT CONFIG [OTHER]}"
OTHER="${3:-}"
HERE=$(cd "$(dirname "$0")" && pwd)
STRATA_DIR=/home/jim/lab/Strata
if [ -f "$HERE/ab.env" ]; then
    set -a
    . "$HERE/ab.env"
    set +a
fi
WAIT_S=${STRATA_WAIT_S:-480}
if [ -n "$OTHER" ]; then
    (
        i=0
        while [ "$i" -lt "$WAIT_S" ]; do
            oport=$(curl -s -m 5 http://127.0.0.1:8033/running | python3 -c '
import json, sys
for m in json.load(sys.stdin).get("running", []):
    if m["model"] == sys.argv[1]:
        print(m["proxy"].rsplit(":", 1)[1])' "$OTHER" 2>/dev/null || true)
            [ -z "$oport" ] && exit 0                   # not loaded: nothing to unload
            if curl -s -m 5 "http://127.0.0.1:$oport/metrics" | python3 -c '
import json, sys
live = json.load(sys.stdin)["live"]
sys.exit(0 if live.get("state") == "idle" and not live.get("queued") else 1)' 2>/dev/null; then
                curl -s -m 30 -X POST "http://127.0.0.1:8033/api/models/unload/$OTHER" >/dev/null 2>&1
                exit 0
            fi
            sleep 1
            i=$((i + 1))
        done
    ) &
fi
# A llama-swap config reload stops the old Strata and starts this one at once, on the
# same port: the old one is still freeing port and VRAM, so the new one exited
# ("upstream command exited prematurely", 2026-10-01). Wait for it, up to STRATA_WAIT_S
# (default 480, inside llama-swap's 600 s healthCheckTimeout), since the other Strata
# may first finish a running request.
i=0
WAIT_TICKS=$(( WAIT_S * 2 ))
while pgrep -f "serve/server.py --engine strata --config" >/dev/null \
        && [ "$i" -lt "$WAIT_TICKS" ]; do
    sleep 0.5
    i=$((i + 1))
done
# GPUs by UUID: when CONFIG has "gpu_uuids", rewrite its "gpu" (nvidia-smi numbers, which
# Strata's server wants) from the cards' current numbers. The numbering changes whenever a
# card is added, removed or moved (the EPYC upgrade, then a card pulled on 2026-10-30), and
# a stale number put Strata on the wrong card or stopped it starting.
python3 - "$CONFIG" <<'PY' || exit 1
import json, subprocess, sys
path = sys.argv[1]
cfg = json.load(open(path))
uuids = cfg.get("gpu_uuids")
if uuids:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    index = {u.strip(): int(i) for i, u in (l.split(",") for l in out.strip().splitlines())}
    missing = [u for u in uuids if u not in index]
    if missing:
        sys.exit(f"run-strata.sh: GPU(s) not found: {missing}")
    gpu = [index[u] for u in uuids]
    if cfg.get("gpu") != gpu:
        print(f"run-strata.sh: gpu {cfg.get('gpu')} -> {gpu} (from gpu_uuids)", file=sys.stderr)
        cfg["gpu"] = gpu
        tmp = path + ".tmp"
        json.dump(cfg, open(tmp, "w"), indent=1)
        __import__("os").replace(tmp, path)
PY
cd "$STRATA_DIR"
exec "$STRATA_DIR/.venv/bin/python" "$STRATA_DIR/serve/server.py" \
    --engine strata \
    --config "$CONFIG" \
    --host 127.0.0.1 \
    --port "$PORT"
