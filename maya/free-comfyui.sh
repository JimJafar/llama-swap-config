#!/bin/sh
# free-comfyui.sh CMD [ARGS...]  (2026-11-06)
#
# ComfyUI runs on the 3090 + ZOTAC 5060 Ti, the cards maya-glm, maya-deepseek and the 27Bs use,
# and with --highvram it keeps its models on them after a job. Before one of those models
# starts: wait (up to 10 min) for ComfyUI's queue to empty, ask it to unload its models and free
# its memory, wait for that to show, then exec the command. ComfyUI not running: start at once.
# The other direction is ComfyUI's marvin-gpu-guard extension (~/llama-swap/comfyui).
C=http://127.0.0.1:8190
if curl -s -m 3 "$C/queue" >/dev/null 2>&1; then
    i=0
    while [ $i -lt 600 ] && curl -s -m 3 "$C/queue" | grep -q '"queue_running": *\[ *\[' ; do sleep 1; i=$((i + 1)); done
    curl -s -m 5 -X POST "$C/free" -H 'content-type: application/json' \
        -d '{"unload_models": true, "free_memory": true}' >/dev/null 2>&1
    # ComfyUI applies the flags in its worker loop; give it a moment, then until its
    # VRAM use on the two cards is small (or 30 s).
    i=0
    while [ $i -lt 30 ]; do
        sleep 1; i=$((i + 1))
        used=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
            | awk -v p="$(pgrep -f '[m]ain.py --listen 127.0.0.1 --port 8190' | head -1)" \
                  -F', ' '$1==p {s+=$2} END {print s+0}')
        [ "$used" -lt 1500 ] && break
    done
fi
exec "$@"
