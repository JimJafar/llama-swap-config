#!/usr/bin/env python3
"""OpenAI-compatible /v1/audio/transcriptions server on the Intel NPU.

Backed by OpenVINO GenAI's STATIC WhisperPipeline (distil-medium.en int8, OpenVINO IR).
Runs entirely on the NPU (no GPU) so the ASR leg works even while 3-GPU --fit models
hold GPU 0. Managed by llama-swap as the `whisper-npu-asr` entry in the `dictation`
group (see /home/jim/llama-swap/config.yaml).

Required env (set by the llama-swap entry `env`): LD_LIBRARY_PATH must include the
venv's openvino/libs dir (rootless NPU driver lives there). See npu/README.md.

Usage: npu-env/bin/python asr_server.py --port ${PORT} [--model-dir DIR]
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import subprocess
import threading
import wave

import numpy as np
import openvino_genai as ov_genai
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

log = logging.getLogger("npu-asr")

MODEL_DIR_DEFAULT = "/mnt/shared/models/distil-medium.en-npu"
# The static pipeline is the ONLY whisper path that works on the NPU (dynamic fails
# with unordered_map::at). NPUW_* offloads unsupported ops to CPU. See npu/README.md.
STATIC_CFG = {
    "NPU_USE_NPUW": "YES",
    "NPUW_DEVICES": "CPU",
    "NPUW_ONLINE_PIPELINE": "NONE",
    "STATIC_PIPELINE": True,
}

app = FastAPI(title="NPU Whisper ASR")
_pipe: ov_genai.WhisperPipeline | None = None
_lock = threading.Lock()
_ready = False


def decode_audio(data: bytes) -> np.ndarray:
    """Decode any audio (wav/mp3/m4a/flac/...) to 16 kHz mono float32 via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000",
         "-f", "wav", "pipe:1"],
        input=data,
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise ValueError(f"ffmpeg decode failed: {proc.stderr.decode(errors='ignore')[:200]}")
    with wave.open(io.BytesIO(proc.stdout), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def init_pipeline(model_dir: str) -> None:
    global _pipe, _ready
    log.info("loading WhisperPipeline from %s on NPU ...", model_dir)
    _pipe = ov_genai.WhisperPipeline(model_dir, device="NPU", **STATIC_CFG)
    _ready = True
    log.info("NPU ASR ready")


@app.get("/health")
def health():
    if not _ready:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


@app.post("/v1/audio/transcriptions")
def transcribe(
    file: UploadFile = File(...),
    model: str = Form("whisper-npu-asr"),  # noqa: ARG001 - llama-swap routes by this
    language: str | None = Form(None),  # noqa: ARG001 - distil .en is English-only
    response_format: str = Form("json"),
    prompt: str | None = Form(None),  # noqa: ARG001
):
    if not _ready:
        return JSONResponse({"error": "ASR model not ready"}, status_code=503)
    try:
        audio = decode_audio(file.file.read())
    except Exception as e:  # noqa: BLE001
        log.warning("decode failed: %s", e)
        return JSONResponse({"error": f"audio decode failed: {e}"}, status_code=400)
    if audio.size == 0:
        return JSONResponse({"error": "empty audio"}, status_code=400)
    pipe = _pipe
    if pipe is None:
        return JSONResponse({"error": "ASR model not ready"}, status_code=503)
    try:
        with _lock:
            out = pipe.generate(audio, max_new_tokens=448)
    except Exception as e:  # noqa: BLE001
        log.exception("generate failed")
        return JSONResponse({"error": str(e)}, status_code=500)
    text = out.strip() if isinstance(out, str) else str(out)
    log.info("transcribed %.1fs audio -> %d chars", audio.size / 16000.0, len(text))
    if response_format == "text":
        return PlainTextResponse(text)
    return {"text": text}


def main() -> None:
    ap = argparse.ArgumentParser()
    _default_port = os.environ.get("PORT", "8091")
    try:
        _default_port = int(_default_port)
    except ValueError:
        _default_port = 8091
    ap.add_argument("--port", type=int, default=_default_port)
    ap.add_argument("--model-dir", default=os.environ.get("NPU_ASR_MODEL_DIR", MODEL_DIR_DEFAULT))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    init_pipeline(args.model_dir)

    import uvicorn

    # 0.0.0.0 is deliberate and matches the other dictation entries: llama-swap proxies
    # to the upstream from localhost, and the host is only reachable over Tailscale.
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
