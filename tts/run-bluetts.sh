#!/usr/bin/env bash
# BlueTTS (BlueV3, v3.0.1) on the CPU, run by llama-swap as `bluetts`: OpenAI-compatible
# POST /v1/audio/speech from BlueTTS's own examples/openai_server.py. No GPU.
# Usage: run-bluetts.sh <port>
#
# Rebuildable parts live in ~/lab/bluetts:
#   code/          git clone https://github.com/maxmelichov/BlueTTS (main), `uv sync --extra server`,
#                  then `uv pip install --no-index --find-links ../bluev3_onnx/frontend polyg2p==0.1.0`
#   bluev3_onnx/   notmax123/BlueV3-onnx at the revision in ~/lab/bluetts/revision, English set only:
#                  onnx/, voices/, frontend/polyg2p_data/ + the manylinux polyg2p wheel, MANIFEST.json
#                  (urls.txt there is the aria2c input list). No frontend/renikud (1.2 GB, Hebrew G2P)
#                  and no frontend/yi, so Hebrew and Yiddish input fail; en/de/es/it/ru work.
# bluetts/blue-release.patch from the bundle is NOT applied: it no longer applies to main,
# which reads the bundle's blue_release section itself.
#
# ORT_NUM_THREADS: BlueTTS defaults to min(8, cores); 8 measured ~12x real time on the EPYC.
cd /home/jim/lab/bluetts/code || exit 1
export ORT_NUM_THREADS="${ORT_NUM_THREADS:-8}"
exec .venv/bin/python examples/openai_server.py \
  --bundle /home/jim/lab/bluetts/bluev3_onnx \
  --host 127.0.0.1 --port "${1:?port}" \
  --default-voice daniel
