#!/bin/sh
# yield-strata.sh <command...>: unload Strata, then run the command (exec, so llama-swap
# still manages the model's own process).
#
# For llama-swap entries that put a lot of the model in system RAM (Flint, the gemma-26B
# entries). Strata pins most of the 62 GB, and loading one of these beside it ran Marvin
# out of RAM: page allocation failures all day on 2026-10-25, then a hard freeze at 14:03.
# Strata's group is persistent, so llama-swap never evicts it on its own; this asks for
# the unload. Going back is automatic: Strata's group is exclusive, so the next Strata
# request evicts these non-persistent models.
#
# The unload request runs in the background: llama-swap may hold its lock until this model
# is up, so the script waits up to 60 s for Strata's process to exit and then starts anyway.
curl -s -m 120 -X POST http://127.0.0.1:8033/api/models/unload/Strata-IQ3S >/dev/null 2>&1 &
i=0
while pgrep -x strata >/dev/null && [ $i -lt 60 ]; do sleep 1; i=$((i + 1)); done
exec "$@"
