#!/bin/sh
# free-strata.sh CMD [ARGS...]: unload Strata, wait for its engine to exit, then exec CMD.
#
# Prefixed to the cmd of every llama-swap entry that needs the 3090 or 5070 Ti (the 27Bs,
# Q3.8-FN, muse-glimmer, gemma-31B). Strata's group is persistent, so llama-swap never
# evicts it, and those models used to fail to load while it was up; Strata stays
# persistent (so models on other cards can't evict it) and these entries free the cards
# themselves, as run-maya.sh does. Strata comes back on its next request, and its
# exclusive group then evicts whichever of these is loaded.
#
# The unload runs in the background: llama-swap may hold its lock until this model is up.
curl -s -m 120 -X POST http://127.0.0.1:8033/api/models/unload/Strata-IQ3S >/dev/null 2>&1 &
i=0
while pgrep -f "Strata/engine/strata --serve" >/dev/null && [ $i -lt 60 ]; do sleep 1; i=$((i + 1)); done
exec "$@"
