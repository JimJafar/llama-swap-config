#!/bin/sh
# Strata launcher for llama-swap:  run-strata.sh PORT CONFIG [OTHER]
#
# Runs the official Strata checkout (~/lab/Strata: its serve/server.py, .venv and
# engine) with one of the engine configs here (strata-iq3_s.json, strata-iq3_xxs.json).
#
# OTHER: the llama-swap model that uses the same two cards (Strata-IQ3S and
# Strata-IQ3XXS need the 5070 Ti + 3090). Both are persistent, so llama-swap never
# evicts one for the other; this unloads OTHER itself, but only once it is idle (no
# request running or queued), up to STRATA_WAIT_S: llama-swap's unload API stops a
# model at once, which killed requests mid-answer.
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
cd "$STRATA_DIR"
exec "$STRATA_DIR/.venv/bin/python" "$STRATA_DIR/serve/server.py" \
    --engine strata \
    --config "$CONFIG" \
    --host 127.0.0.1 \
    --port "$PORT"
