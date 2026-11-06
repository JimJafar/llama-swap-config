#!/bin/sh
# free-models.sh MODEL[,MODEL...] CMD [ARGS...]: unload these llama-swap models, wait until
# llama-swap no longer lists them as running, then exec CMD.
#
# Prefixed to the cmd of entries that need cards held by models llama-swap won't evict
# itself: the default pair (Basalt-QFN on the 5070 Ti, maya-glm on the 3090 + MSI) is in a
# persistent group, so nothing else can evict it, and Strata was before it (2026-11-01;
# this replaced free-strata.sh). The pair comes back on its next request, and its
# exclusive group then evicts whichever of these is loaded.
#
# The unloads run in the background: llama-swap may hold its lock until this model is up.
MODELS="${1:?usage: free-models.sh MODEL[,MODEL...] CMD [ARGS...]}"
shift
for m in $(echo "$MODELS" | tr ',' ' '); do
    curl -s -m 120 -X POST "http://127.0.0.1:8033/api/models/unload/$m" >/dev/null 2>&1 &
done
i=0
while [ $i -lt 90 ]; do
    running=$(curl -s -m 5 http://127.0.0.1:8033/running 2>/dev/null)
    busy=""
    for m in $(echo "$MODELS" | tr ',' ' '); do
        echo "$running" | grep -q "\"model\":\"$m\"" && busy=1
    done
    [ -z "$busy" ] && break
    sleep 1; i=$((i + 1))
done
# llama-swap drops a model from /running before its engine has exited and freed the VRAM
# (the first 27B start after this script went in failed that way), so also wait for the
# engines' own processes. The [x] patterns can't match this script's own command line.
# FREE_MODELS_NO_ENGINE_WAIT=1 (Basalt-Q4KXL, 2026-11-06): the models it frees are llama.cpp
# containers, and the engines this loop waits for (GLM on the other two cards) are not in its way.
[ -n "${FREE_MODELS_NO_ENGINE_WAIT:-}" ] && i=120
while [ $i -lt 120 ] && pgrep -f '[S]trata/engine/strata --serve|[b]in/basalt --serve|[b]uild/maya --serve|[b]uild/maya ' >/dev/null; do
    sleep 1; i=$((i + 1))
done
sleep 2
exec "$@"
