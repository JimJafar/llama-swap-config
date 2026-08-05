# NPU Speech-to-Text (dictation ASR experiment)

Run Whisper-family ASR on the Core Ultra 7 265KF's **NPU 3** (13 TOPS, `/dev/accel/accel0`)
via OpenVINO GenAI's *static* whisper pipeline — no GPU involved. Built 2026-08-04 as an
experiment to exercise the otherwise-idle NPU and evaluate it against Parakeet for dictation.

## Result summary (7 real clips, 3–29 s, LibriSpeech test-clean + JFK, normalized WER)

| Model (int8, OpenVINO IR) | Size | AVG WER | 6 s clip | 30 s clip | 62 s clip |
|---|---|---|---|---|---|
| `whisper-base-int8-ov` (Intel pre-converted) | 81 MB | 0.062 | 0.29 s | 0.53 s | — |
| `distil-whisper/distil-small.en` (our export) | 247 MB | 0.104 | 0.72 s | 1.10 s | — |
| `distil-whisper/distil-medium.en` (our export) | 476 MB | **0.050** | 1.51 s | 1.73 s | 5.0 s (12.5× realtime) |

All clips transcribe **faster than real-time** on the 13-TOPS NPU. Long audio (>30 s) is
chunked automatically (sliding window) — verified on a 62.5 s concatenation.

**Takeaways**

- **distil-medium.en is the quality pick on the NPU** (best WER of the three; closest to
  Parakeet's ~1.4% LibriSpeech test-clean figure). It even nailed the JFK quote that
  distil-small dropped a phrase from.
- **distil-small.en is not worth it here**: WER was *no better* than whisper-base and it is
  ~3× slower (its encoder is whisper-small's, bigger than base's).
- **whisper-base is the speed pick** (~0.25 s/clip, 60× realtime) if quality can slip.
- Silent audio makes Whisper-family models hallucinate short text (e.g. `"You're not going
  to."`); Parakeet returns empty. Only matters if silence reaches the ASR.

## Why this was non-trivial (gotchas, all solved)

1. **No user-space NPU driver on the box.** Kernel driver (`intel_vpu`) + firmware were
   present, but OpenVINO's NPU plugin also needs the Level Zero loader + Intel NPU driver.
   Installed **rootless** by dropping `libze_loader.so.1.32.0` (Arch `level-zero-loader`
   pkg) + `libze_intel_npu.so.1.35.0` + `libopenvino_intel_npu_compiler*.so` (from
   github.com/intel/linux-npu-driver v1.35.0 release debs) into
   `npu-env/lib/python3.13/site-packages/openvino/libs/`, plus a
   `libnpu_driver_compiler.so -> libopenvino_intel_npu_compiler.so` symlink. **Every run
   needs `LD_LIBRARY_PATH=<that dir>`** (the loader does not self-locate).
2. **The dynamic `WhisperPipeline` does not work on NPU.** Must use the static pipeline:

   ```python
   config = {"NPU_USE_NPUW": "YES", "NPUW_DEVICES": "CPU",
             "NPUW_ONLINE_PIPELINE": "NONE", "STATIC_PIPELINE": True}
   pipe = ov_genai.WhisperPipeline(model_dir, "NPU", **config)
   ```

3. **Export recipe matters.** `optimum-cli`/`OVModelForSpeechSeq2Seq.from_pretrained` with
   transformers 4.51.3 constant-folds the causal mask → the static pipeline's
   `add_attention_mask_input` patcher can't find its `Range→Convert→Greater→Convert`
   pattern → `Port for tensor name attention_mask was not found`. With
   **transformers 4.50.2 + optimum-intel 1.26.0** the mask stays symbolic
   (`Range×2, Greater×1`), and then one **graph surgery** is still needed (insert an
   identity `Convert(Range)` before `Greater` so pattern 1 matches) — see
   `patch_decoder.py`. After patching, the pipeline replaces the mask subgraph with the
   `attention_mask` input and it just works.
4. Python 3.14 cannot host this stack (openvino pins numpy<2.3, no cp314 wheels) — the
   venv is **Python 3.13**. torch must be **2.7.1** (torch 2.9+ removed the legacy
   `torch.onnx.symbolic_opset14` helpers optimum's exporter imports).

## Setup (already done on Marvin; for rebuilds)

```bash
uv venv npu-env --python python3.13
uv pip install --python npu-env/bin/python \
  openvino==2026.3.0 openvino-tokenizers openvino-genai \
  optimum-intel==1.26.0 transformers==4.50.2 nncf==2.18.0 torch==2.7.1 pyarrow
# rootless NPU driver: see gotcha 1 — copy the 5 .so/symlinks into openvino/libs/
```

## Export + patch + run

```bash
export HF_HOME=$HOME/.cache/huggingface
npu-env/bin/optimum-cli export openvino --trust-remote-code \
  --model distil-whisper/distil-medium.en --weight-format int8 \
  /mnt/shared/models/distil-medium.en-npu
# one-line graph surgery on the decoder (REQUIRED for the static NPU pipeline):
npu-env/bin/python npu/patch_decoder.py \
  /mnt/shared/models/distil-medium.en-npu/openvino_decoder_model.xml \
  /mnt/shared/models/distil-medium.en-npu/openvino_decoder_model_patched.xml
# (then move patched over original)
LD_LIBRARY_PATH=<openvino/libs> npu-env/bin/python npu/bench_whisper.py \
  /mnt/shared/models/distil-medium.en-npu
```

Intel's own pre-converted models (`OpenVINO/whisper-base-int8-ov` etc., no patching
needed) live in `/mnt/shared/models/`; the `OpenVINO/` HF org also publishes
whisper-medium/large and distil-whisper-large int4/int8 variants.

## Next step (not done — needs Jim's go-ahead)

Wire the winner into the dictation pipeline: a small FastAPI OpenAI-compatible
`/v1/audio/transcriptions` server over `WhisperPipeline(..., "NPU")`, registered in
llama-swap as a `dictation` group member in place of / alongside `parakeet-asr`, and the
`stt-warmup` script updated to warm the NPU model instead of GPU 0. That fully frees the
ZOTAC 5060 Ti (GPU 0) so dictation can run during 3-GPU fit models.
