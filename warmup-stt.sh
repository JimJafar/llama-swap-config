#!/bin/sh
# STT preload (added 2026-07-28; ASR target moved to the NPU 2026-08-04; TTS added
# 2026-09-12). Warms the resident voice stack on the ZOTAC after llama-swap comes up:
# the NPU ASR server (whisper-npu-asr, no GPU) and the TTS server (chatterbox-turbo,
# ~1.6 GiB). Registered as a PM2 app; it idles after warming so it survives
# `pm2 resurrect` on reboot and re-warms each boot.
#
# qwen-clean-2b is NO LONGER warmed (2026-09-12). s1-mini replaced it as baruch-server's
# cleanup model, so it is now an `unlisted` on-demand fallback that must not hold ZOTAC
# VRAM at boot. gemma-4-E4B-MTP and s1-mini are both CHAT models, so llama-swap's own
# `preload:` in config.yaml warms those two directly -- only the non-chat backends need
# warming from here.
#
# WHY THIS EXISTS FOR ASR/TTS (2026-09-12): llama-swap's own `preload:` DOES work, but it
# warms a preloaded model with a CHAT-shaped request. A chat backend answers it; an ASR or
# TTS backend returns 404, and llama-swap then STOPS the process it just started --
# observably `[ERROR] failed to preload model ...: status 404` followed by
# `<model> upstream process exited unexpectedly`. So non-chat backends must be loaded
# through their real endpoint. gemma-4-E4B-MTP and s1-mini are chat models and are warmed
# by `preload:` in config.yaml; whisper-npu-asr and chatterbox-turbo are warmed here.
#
# NB: this only fires at boot / `pm2 (re)start stt-warmup`. After a bare `pm2 restart llama-swap`
# (without a reboot), re-run `pm2 restart stt-warmup` too, or these will load lazily on first use.
E=http://127.0.0.1:8033
echo "[stt-warmup] waiting for llama-swap on $E ..."
until curl -sf "$E/v1/models" >/dev/null 2>&1; do sleep 3; done
# ASR (whisper-npu-asr on the NPU): llama-swap routes by the multipart `model` field and
# starts the model before proxying, so the tiny silent WAV just triggers the load (the
# transcription result, a Whisper-family silence hallucination, is discarded).
curl -s "$E/v1/audio/transcriptions" -F 'model=whisper-npu-asr' \
	-F 'file=@/home/jim/llama-swap/warmup-silence.wav;type=audio/wav' \
	>/dev/null 2>&1 && echo "[stt-warmup] whisper-npu-asr loaded"
# TTS (chatterbox-turbo, ZOTAC): same trick via the speech endpoint. Omit `voice` -- a
# `voice` field is read as a reference-WAV path and fails unless --voice-dir has it.
warm_tts() {
	curl -s -m 180 "$E/v1/audio/speech" -H 'Content-Type: application/json' \
		-d '{"model":"chatterbox-turbo","input":"warmup","response_format":"wav"}' \
		-o /dev/null 2>&1 && echo "[stt-warmup] chatterbox-turbo loaded"
}
warm_tts

# --- KEEP-ALIVE -------------------------------------------------------------------
# Added 2026-09-12. A one-shot warm is NOT enough to keep chatterbox-turbo "always on":
# every `-watch-config` reload makes llama-swap stop the gemma-resident group's models
# and then restart only the ones it can warm via `preload:` -- which excludes TTS/ASR
# backends (see above), so chatterbox is stopped and never brought back. Editing
# config.yaml therefore silently took TTS down until this loop existed.
# chat models are unaffected: llama-swap restarts gemma-4-E4B-MTP / s1-mini itself.
# Liveness is checked via the process table (cheap, no audio generated); the re-warm is
# a single request through the real endpoint. Bracket in the pattern so pgrep does not
# match its own command line.
while true; do
	sleep 60
	if ! pgrep -f "[c]rispasr --server" >/dev/null 2>&1; then
		echo "[stt-warmup] chatterbox-turbo is down; re-warming"
		warm_tts
	fi
done
