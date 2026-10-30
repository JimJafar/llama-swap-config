#!/bin/sh
# Project Maya under llama-swap:  run-maya.sh PORT CONFIG UUID [UUID ...]
#
# CONFIG is one of Maya's own configs in ~/lab/project-maya (maya-maya-l.json for
# GLM-5.3-Flash, maya-ds4.json for DeepSeek-V4.1-Flash-1017). The UUIDs are the GPUs
# to split the layers across, in order. Cards that aren't installed are skipped, so it
# runs on whatever is in the machine (e.g. three cards while the ZOTAC is out).
#
# 1. Rewrites the config's "gpu" (nvidia-smi numbers, which Maya wants) from the UUIDs:
#    the numbering changes whenever a card is added, removed or moved.
# 2. Unloads Strata: Maya needs every card, and Strata's group is persistent, so
#    llama-swap won't evict it. Going back is automatic: a Strata request evicts Maya
#    (Strata's group is exclusive, the "maya" group isn't persistent).
# 3. Runs Maya's dashboard + API on PORT. Maya stops its server and engine itself on the
#    SIGTERM llama-swap sends.
#
# Maya pins most of the free RAM while it runs (all but ~6 GB).
set -eu
PORT="${1:?usage: run-maya.sh PORT CONFIG UUID...}"
CONFIG="${2:?usage: run-maya.sh PORT CONFIG UUID...}"
shift 2
MAYA=/home/jim/lab/project-maya

python3 - "$CONFIG" "$@" <<'PY'
import json, os, subprocess, sys
path, uuids = sys.argv[1], sys.argv[2:]
out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                     capture_output=True, text=True, check=True).stdout
index = {u.strip(): int(i) for i, u in (l.split(",") for l in out.strip().splitlines())}
gpu = [index[u] for u in uuids if u in index]
skipped = [u for u in uuids if u not in index]
if not gpu:
    sys.exit("run-maya.sh: none of the GPUs are installed")
cfg = json.load(open(path))
print(f"run-maya.sh: gpu {cfg.get('gpu')} -> {gpu}" + (f" (not installed: {skipped})" if skipped else ""),
      file=sys.stderr)
if cfg.get("gpu") != gpu:
    cfg["gpu"] = gpu
    json.dump(cfg, open(path + ".tmp", "w"), indent=1)
    os.replace(path + ".tmp", path)
PY

# Unload Strata in the background (llama-swap may hold its lock until this model is up),
# then wait up to 60 s for its engine to exit so Maya sees free cards.
curl -s -m 120 -X POST http://127.0.0.1:8033/api/models/unload/Strata-IQ3S >/dev/null 2>&1 &
i=0
while pgrep -f "Strata/engine/strata --serve" >/dev/null && [ $i -lt 60 ]; do sleep 1; i=$((i + 1)); done

cd "$MAYA"
exec "$MAYA/.venv/bin/python" "$MAYA/maya.py" --config "$CONFIG" --port "$PORT"
