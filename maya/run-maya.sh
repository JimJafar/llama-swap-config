#!/bin/sh
# Project Maya under llama-swap:  run-maya.sh PORT CONFIG [--max-context N] UUID [UUID ...]
#
# CONFIG is one of Maya's own configs in ~/lab/project-maya (maya-maya-l.json for
# GLM-5.3-Flash, maya-ds4.json for DeepSeek-V4.1-Flash-1017). The UUIDs are the GPUs
# to split the layers across, in order. Cards that aren't installed are skipped, so it
# runs on whatever is in the machine (e.g. three cards while the ZOTAC is out).
#
# 1. Rewrites the config's "gpu" (nvidia-smi numbers, which Maya wants) from the UUIDs:
#    the numbering changes whenever a card is added, removed or moved. With
#    --max-context N, also sets the engine's context there (so llama-swap's config.yaml is
#    where it's chosen, and sync-models.py reads it for pi's models.json). Then merges in
#    maya/<CONFIG name>.overrides.json when there is one ("env" keys, "args" flag values):
#    tuned settings kept here, since Maya's setup rewrites its config on an update.
# 2. Unloads Strata: Maya needs every card, and Strata's group is persistent, so
#    llama-swap won't evict it. Going back is automatic: a Strata request evicts Maya
#    (Strata's group is exclusive, the "maya" group isn't persistent).
# 3. Runs Maya's dashboard + API on PORT. Maya stops its server and engine itself on the
#    SIGTERM llama-swap sends.
#
# Maya pins most of the free RAM while it runs (all but ~6 GB; not maya-ik2, which maps its file).
#
# MAYA_DIR (env) picks another Maya checkout: inkling-2-mini runs from a git worktree at
# ~/lab/maya-ik2 (newer Maya, built by hand; see config.yaml).
set -eu
PORT="${1:?usage: run-maya.sh PORT CONFIG UUID...}"
CONFIG="${2:?usage: run-maya.sh PORT CONFIG UUID...}"
shift 2
CTX=""
if [ "${1:-}" = "--max-context" ]; then CTX="$2"; shift 2; fi
MAYA="${MAYA_DIR:-/home/jim/lab/project-maya}"

MAYA_OVERRIDES_DIR="$(cd "$(dirname "$0")" && pwd)" python3 - "$CONFIG" "$CTX" "$@" <<'PY'
import json, os, subprocess, sys
path, ctx, uuids = sys.argv[1], sys.argv[2], sys.argv[3:]
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
changed = cfg.get("gpu") != gpu
cfg["gpu"] = gpu
if ctx:
    args = cfg.get("args", [])
    if "--max-context" in args:
        i = args.index("--max-context")
        if args[i + 1] != ctx:
            print(f"run-maya.sh: --max-context {args[i + 1]} -> {ctx}", file=sys.stderr)
            args[i + 1] = ctx
            changed = True
    else:
        args += ["--max-context", ctx]
        changed = True
# Tuned settings kept in this repo (maya/<config name>.overrides.json): Maya's setup
# rewrites its config on an update, so they are merged in again at every start.
over_path = os.path.join(os.environ["MAYA_OVERRIDES_DIR"],
                         os.path.basename(path).replace(".json", ".overrides.json"))
if os.path.exists(over_path):
    over = json.load(open(over_path))
    env = cfg.setdefault("env", {})
    for k, v in over.get("env", {}).items():
        if env.get(k) != v:
            print(f"run-maya.sh: env {k} {env.get(k)} -> {v}", file=sys.stderr)
            env[k] = v
            changed = True
    args = cfg.setdefault("args", [])
    for flag, v in over.get("args", {}).items():
        if flag in args:
            i = args.index(flag)
            if args[i + 1] != v:
                print(f"run-maya.sh: {flag} {args[i + 1]} -> {v}", file=sys.stderr)
                args[i + 1] = v
                changed = True
        else:
            print(f"run-maya.sh: {flag} {v} (added)", file=sys.stderr)
            args += [flag, v]
            changed = True
# Clamp an over-long max_tokens instead of refusing it (2026-11-21, as Basalt's fit-max-tokens):
# pi asks for the context left after its own estimate of the prompt, which runs under Maya's
# count, so prompt + max_tokens overshot and Maya answered 400 context_length_exceeded.
if cfg.get("fit_max_tokens") is not True:
    print("run-maya.sh: fit_max_tokens -> true", file=sys.stderr)
    cfg["fit_max_tokens"] = True
    changed = True
if changed:
    json.dump(cfg, open(path + ".tmp", "w"), indent=1)
    os.replace(path + ".tmp", path)
PY

# Unload Strata in the background (llama-swap may hold its lock until this model is up),
# then wait up to 60 s for its engine to exit so Maya sees free cards.
curl -s -m 120 -X POST http://127.0.0.1:8033/api/models/unload/Strata-IQ3S >/dev/null 2>&1 &
i=0
while pgrep -f "Strata/engine/strata --serve" >/dev/null && [ $i -lt 60 ]; do sleep 1; i=$((i + 1)); done

# On a config reload llama-swap starts this before the previous Maya engine has exited and
# freed its cards (2026-11-01: the reloaded pair failed to start); wait up to 60 s more.
i=0
while pgrep -f '[b]uild/maya --serve|[b]uild/maya-ds4 |[b]uild/maya-ik2 ' >/dev/null && [ $i -lt 120 ]; do sleep 1; i=$((i + 1)); done
# 2026-11-06: and for the old server to let go of the port. On a reload llama-swap reuses the
# port, and Maya's server outlives its engine by a few seconds ("port 5915 is already in use").
i=0
while ss -Htln "sport = :$PORT" | grep -q . && [ $i -lt 60 ]; do sleep 1; i=$((i + 1)); done

cd "$MAYA"
exec "$MAYA/.venv/bin/python" "$MAYA/maya.py" --config "$CONFIG" --port "$PORT"
