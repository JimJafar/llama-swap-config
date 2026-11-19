#!/usr/bin/env python3
"""OpenAI-compatible /v1/audio/transcriptions server on the CPU, backed by transcribe.cpp.

Replaces npu/asr_server.py (the Intel NPU went with the Core Ultra). Serves any
transcribe.cpp GGUF; llama-swap runs it with Parakeet TDT 0.6B v2 as the `parakeet-cpu`
entry (see /home/jim/llama-swap/config.yaml). Why that model: bench/asr/README.md.

Needs TRANSCRIBE_LIBRARY pointing at a shared libtranscribe (a CPU-only build:
cmake -B build-cpu-shared -DTRANSCRIBE_BUILD_SHARED=ON -DGGML_CUDA=OFF -DGGML_NATIVE=ON)
and a venv with the transcribe.cpp Python bindings, fastapi, uvicorn, python-multipart, numpy.

Usage: stt-env/bin/python asr_server.py --port PORT --model MODEL.gguf [--threads N]
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import threading
import time

import numpy as np
import transcribe_cpp
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

log = logging.getLogger("cpu-asr")
app = FastAPI(title="CPU ASR (transcribe.cpp)")
_session = None
_model_name = "asr"
_lock = threading.Lock()


def decode_audio(data: bytes) -> np.ndarray:
    """Decode any audio (wav/mp3/m4a/flac/ogg/webm...) to 16 kHz mono float32 via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000",
         "-f", "f32le", "pipe:1"],
        input=data, capture_output=True,
    )
    if proc.returncode != 0:
        raise ValueError(proc.stderr.decode(errors="ignore")[:200] or "ffmpeg failed")
    return np.frombuffer(proc.stdout, dtype=np.float32)


@app.get("/health")
def health():
    return {"status": "ok"} if _session is not None else JSONResponse({"status": "loading"}, 503)


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": _model_name, "object": "model", "owned_by": "local"}]}


@app.post("/v1/audio/transcriptions")
def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(""),  # noqa: ARG001 - llama-swap routes by this
    language: str | None = Form(None),  # noqa: ARG001 - Parakeet v2 is English-only
    prompt: str | None = Form(None),  # noqa: ARG001 - Parakeet takes no prompt
    response_format: str = Form("json"),
    temperature: float | None = Form(None),  # noqa: ARG001 - greedy decoding
):
    if response_format not in ("json", "text", "verbose_json"):
        return JSONResponse({"error": f"response_format {response_format!r} not supported "
                                      "(json, text, verbose_json)"}, 400)
    try:
        pcm = decode_audio(file.file.read())
    except ValueError as e:
        return JSONResponse({"error": f"audio decode failed: {e}"}, 400)
    if pcm.size == 0:
        return JSONResponse({"error": "empty audio"}, 400)
    t0 = time.monotonic()
    try:
        with _lock:
            result = _session.run(pcm)
    except Exception as e:  # noqa: BLE001
        log.exception("transcription failed")
        return JSONResponse({"error": str(e)}, 500)
    text = (result.text or "").strip()
    dur = pcm.size / 16000.0
    log.info("%.1f s audio -> %d chars in %.2f s", dur, len(text), time.monotonic() - t0)
    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "verbose_json":
        segs = [{"id": i, "start": s.t0_ms / 1000, "end": s.t1_ms / 1000, "text": s.text.strip()}
                for i, s in enumerate(result.segments or [])]
        return {"task": "transcribe", "language": result.language or "en", "duration": dur,
                "text": text, "segments": segs}
    return {"text": text}


def main() -> None:
    global _session, _model_name
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", required=True, help="transcribe.cpp GGUF")
    ap.add_argument("--name", default="asr", help="model id reported by /v1/models")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    _model_name = args.name
    m = transcribe_cpp.Model(args.model, backend="cpu")
    _session = m.session(n_threads=args.threads)
    _session.run(np.zeros(16000, dtype=np.float32))  # warm the graph caches
    log.info("ready: %s, %d threads", args.model, args.threads)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
