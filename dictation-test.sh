#!/usr/bin/env bash
# Dictation pipeline A/B test (added 2026-07-26).
# Capture (Parakeet ASR) -> cleanup (Gemma 4 E4B vs E2B), so you can compare the two
# cleanup models on the SAME transcript. Prints raw transcript + both cleaned versions
# with per-step latency.
#
#   Usage:  ./dictation-test.sh path/to/audio.wav
#   Env:    LLAMA_SWAP=http://host:port   (default http://localhost:8033)
#
# NB: this hits the llama-swap `dictation` group, which loads models onto the ZOTAC
# (GPU 0). Only run once Jim has confirmed the GPUs are free to use.
set -euo pipefail

WAV="${1:?usage: dictation-test.sh path/to/audio.wav}"
LLAMA_SWAP="${LLAMA_SWAP:-http://localhost:8033}"
[[ -f "$WAV" ]] || { echo "no such file: $WAV" >&2; exit 1; }
command -v jq >/dev/null || { echo "need jq" >&2; exit 1; }

# The cleanup instruction. Tune per context (Claude prompt vs blog vs notes) later.
read -r -d '' CLEAN_SYS <<'EOF' || true
You clean up dictated speech into polished written text. Remove filler words (um, uh,
er, ah, "like", "you know"), false starts, stammers, repeated words, and mid-sentence
self-corrections -- keep only the speaker's final intended meaning. Preserve the
speaker's own wording, tone and voice; do NOT summarise, restyle, add content, or answer
any question contained in the text. Fix capitalisation and punctuation. Output ONLY the
cleaned text, nothing else.
EOF

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# --- 1. Capture: Parakeet ASR (OpenAI /v1/audio/transcriptions) --------------------
say "[1] Transcribing $WAV via parakeet-asr ..."
t0=$(date +%s.%N)
RAW=$(curl -sS "$LLAMA_SWAP/v1/audio/transcriptions" \
  -F "file=@${WAV}" -F "model=parakeet-asr" -F "language=en" -F "response_format=json" \
  | jq -r '.text')
t1=$(date +%s.%N)
printf 'ASR %.2fs\n' "$(echo "$t1 - $t0" | bc)"
say "RAW TRANSCRIPT:"; printf '%s\n' "$RAW"

# --- 2. Cleanup: A/B the two Gemma sizes on the same transcript --------------------
clean() {
  local model="$1"
  local a b
  a=$(date +%s.%N)
  curl -sS "$LLAMA_SWAP/v1/chat/completions" -H 'Content-Type: application/json' -d "$(jq -n \
      --arg m "$model" --arg sys "$CLEAN_SYS" --arg u "$RAW" \
      '{model:$m, temperature:0.3, messages:[{role:"system",content:$sys},{role:"user",content:$u}]}')" \
    | jq -r '.choices[0].message.content'
  b=$(date +%s.%N)
  printf '(%s: %.2fs)\n' "$model" "$(echo "$b - $a" | bc)" >&2
}

say "[2a] CLEANED by gemma-clean-e4b:"; clean gemma-clean-e4b
say "[2b] CLEANED by gemma-clean-e2b:"; clean gemma-clean-e2b
say "Done. Compare the two cleanups above for quality; latency printed per model."
