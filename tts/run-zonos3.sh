#!/usr/bin/env bash
# ZONOS3-Mini (Zyphra) on the CPU, run by llama-swap as `zonos3`: Zyphra's own C++/ggml
# zonos3-server (servers/ggml of github.com/Zyphra/ZONOS3-Mini). No GPU, no Python at runtime.
# Usage: run-zonos3.sh <port>
#
# Rebuildable parts live in ~/lab/zonos3 (a clone of github.com/Zyphra/ZONOS3-Mini, main):
#   build:   git submodule update --init servers/ggml/ggml third_party/zyphra-textnorm
#            cd servers/ggml && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON && cmake --build build -j
#            The text normaliser is Rust, built by CMake with cargo: rustup lives in ~/lab/rust
#            (RUSTUP_HOME=~/lab/rust/rustup CARGO_HOME=~/lab/rust/cargo, cargo/bin on PATH).
#   models:  models/gguf/ from Zyphra/ZONOS3-Mini at the revision in ~/lab/zonos3/models-revision
#            (urls.txt there is the aria2c list): backbone-ptq.gguf (Q4_0 experts, Q8_0 mixers),
#            zac-keep2.gguf (the fast decoder), spk-encoder.gguf, steer.gguf, voices/English/.
#            Not downloaded: zac.gguf (full-size decoder, 2.2 GB), zac-encoder.gguf + ssl-teacher.gguf
#            (5.5 GB, raw-reference cloning), the speech-edit backbone. The preset voices ship their
#            reference codes, so they need neither encoder.
cd /home/jim/lab/zonos3 || exit 1
exec servers/ggml/build/zonos3-server models/gguf/backbone-ptq.gguf --cpu \
  --host 127.0.0.1 --port "${1:?port}" \
  --tts-default-voices-dir models/gguf/voices
