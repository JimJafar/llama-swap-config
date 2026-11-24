#!/bin/sh
# pin.sh: keep one model's cards for it (2026-11-24, for maya-deepseek-max on all four cards).
#
#   pin.sh on MODEL CMD [ARGS...]  write MODEL into ~/llama-swap/pinned, then exec CMD with
#                                  PIN_OWNER=MODEL (an entry's cmd prefix: the pin is set at start)
#   pin.sh off                     remove the pin: Basalt (`subagent`) and the rest start again
#   pin.sh                         show the pin
#
# While the file names a model, the start scripts of every other GPU model refuse to start
# (pin-check.sh, sourced by strata/free-models.sh, maya/free-comfyui.sh, maya/run-maya.sh and
# basalt/run-basalt.sh), so a request to Basalt-Q4KXL or `subagent` fails at once instead of
# unloading the pinned model; ComfyUI's marvin-gpu-guard leaves it loaded too. The pinned
# model's own group is persistent, so llama-swap never evicts it either.
#
# The pin stays after the model is unloaded (or llama-swap reloads its config, which stops
# it): only `pin.sh off` lifts it. Plain `docker run` entries (bge-m3, the curators, the
# voice models) have no check; they are small and can't evict anything.
PIN_FILE=/home/jim/llama-swap/pinned
case "${1:-}" in
    on)
        MODEL="${2:?usage: pin.sh on MODEL CMD [ARGS...]}"; shift 2
        echo "$MODEL" > "$PIN_FILE"
        export PIN_OWNER="$MODEL"
        exec "$@" ;;
    off)
        if [ -e "$PIN_FILE" ]; then echo "unpinned $(cat "$PIN_FILE")"; rm -f "$PIN_FILE"
        else echo "not pinned"; fi ;;
    "")
        if [ -s "$PIN_FILE" ]; then echo "pinned: $(cat "$PIN_FILE")"; else echo "not pinned"; fi ;;
    *)
        echo "usage: pin.sh [on MODEL CMD... | off]" >&2; exit 2 ;;
esac
