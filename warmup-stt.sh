#!/bin/sh
# STT preload (added 2026-07-28). llama-swap has no preload, and `persistent: true` only KEEPS
# the dictation group resident once loaded -- it does NOT load it at startup. This warms both
# dictation models after llama-swap comes up, so STT claims its ~5.4 GB on GPU0 (the ZOTAC)
# BEFORE any 3-GPU --fit model can, guaranteeing coexistence. Registered as a PM2 app; it idles
# after warming so it survives `pm2 resurrect` on reboot and re-warms each boot.
#
# NB: this only fires at boot / `pm2 (re)start stt-warmup`. After a bare `pm2 restart llama-swap`
# (without a reboot), re-run `pm2 restart stt-warmup` too, or STT will load lazily on first dictation.
E=http://127.0.0.1:8033
echo "[stt-warmup] waiting for llama-swap on $E ..."
until curl -sf "$E/v1/models" >/dev/null 2>&1; do sleep 3; done
# cleanup LLM (OpenAI chat endpoint)
curl -s "$E/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen-clean-2b","messages":[{"role":"user","content":"warmup"}],"max_tokens":1}' \
  >/dev/null 2>&1 && echo "[stt-warmup] qwen-clean-2b loaded"
# ASR (parakeet): llama-swap routes by the multipart `model` field and starts the model before
# proxying, so the tiny silent WAV just triggers the load (the transcription result is discarded).
curl -s "$E/v1/audio/transcriptions" -F 'model=parakeet-asr' \
  -F 'file=@/home/jim/llama-swap/warmup-silence.wav;type=audio/wav' \
  >/dev/null 2>&1 && echo "[stt-warmup] parakeet-asr loaded"
echo "[stt-warmup] done; idling to stay resurrectable"
sleep infinity
