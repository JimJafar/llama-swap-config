#!/usr/bin/env bash
# CPU-only ASR benchmark (2026-11-19): every Q8_0 GGUF in ~/lab/transcribe.cpp/models through
# transcribe.cpp's CPU build, over the LibriSpeech test-other sample in refs.tsv.
# Usage: bench.sh [threads]   (default 8). Needs ~/lab/asr-bench/audio/<id>.wav (see README.md).
HERE=$(cd "$(dirname "$0")" && pwd)
A=~/lab/asr-bench; T=${1:-8}; mkdir -p "$A/results"
ls "$A"/audio/*.wav > "$A/list.txt"
for g in ~/lab/transcribe.cpp/models/*/*-Q8_0.gguf; do
  n=$(basename "$g" -Q8_0.gguf); out="$A/results/$n-t$T.jsonl"
  ~/lab/transcribe.cpp/build-cpu/bin/transcribe-cli -q --backend cpu --threads "$T" -l en -m "$g" \
    --batch "$A/list.txt" --batch-jsonl > "$out" 2> "${out%.jsonl}.err"
  echo "{\"model\": \"$n\", \"threads\": $T, \"score\": $(~/lab/npu-env/bin/python -I "$HERE/score.py" "$HERE/refs.tsv" "$out")}"
done
