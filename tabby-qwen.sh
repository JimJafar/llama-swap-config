#!/usr/bin/bash
set -euo pipefail

if [[ $# -ne 3 || ! $1 =~ ^[0-9]+$ || $2 != --max-seq-len || ! $3 =~ ^[0-9]+$ ]]; then
  echo "usage: $0 PORT --max-seq-len TOKENS" >&2
  exit 64
fi

umask 077
cd /home/jim/code/tabbyAPI
export TABBY_NETWORK_PORT="$1"
export TABBY_MODEL_MAX_SEQ_LEN="$3"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="\
GPU-93ab52e5-8e61-197e-880a-aa6af5a13c74,\
GPU-d64df8cf-c877-b43d-d9a5-d6b0e38e504a,\
GPU-57bbb375-a61a-6662-e9c8-447b948f0833"
exec .venv/bin/python main.py \
  --config /home/jim/llama-swap/tabbyapi-qwen38-exl3.yml
