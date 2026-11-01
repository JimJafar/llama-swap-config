#!/bin/sh
# Basalt (Qwen3.8-Flash-Next) under llama-swap:  run-basalt.sh PORT CONFIG GPU-UUID
#
# The card is named by UUID (nvidia-smi numbering changes whenever a card is added or
# pulled): CUDA_VISIBLE_DEVICES makes it the only card the process sees, and the config's
# "gpu": "0" is that card. Basalt is in ~/lab/basalt (built from github.com/jesdga95/basalt),
# with its own .venv; the config's --port is overridden by llama-swap's.
PORT="${1:?usage: run-basalt.sh PORT CONFIG GPU-UUID}"
CONFIG="${2:?usage: run-basalt.sh PORT CONFIG GPU-UUID}"
UUID="${3:?usage: run-basalt.sh PORT CONFIG GPU-UUID}"
export CUDA_VISIBLE_DEVICES="$UUID"
# On a config reload llama-swap starts this before the previous Basalt has exited and freed
# the card; wait (up to 60 s) for it. The [b] pattern can't match this script itself.
i=0
while pgrep -f '[b]in/basalt --serve' >/dev/null && [ $i -lt 60 ]; do sleep 1; i=$((i + 1)); done
cd "$HOME/lab/basalt" || exit 1
exec bin/basalt serve --config "$CONFIG" --port "$PORT"
