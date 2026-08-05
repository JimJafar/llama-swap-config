# NPU Speech-to-Text (dictation ASR experiment)

Run Whisper-family ASR on the Core Ultra 7 265KF's **NPU 3** (13 TOPS, `/dev/accel/accel0`)
via OpenVINO GenAI's *static* whisper pipeline — no GPU involved. Built 2026-08-04 as an
experiment to exercise the otherwise-idle NPU and evaluate it against Parakeet for dictation.

## Result summary (7 real clips, 3–29 s, LibriSpeech test-clean + JFK, normalized WER)

### On the NPU (int8, OpenVINO IR, static pipeline)

| Model | Size | AVG WER | 6 s clip | 30 s clip | Load |
|---|---|---|---|---|---|
| **distil-medium.en** (chosen) | 476 MB | **0.050** | 1.51 s | 1.73 s | 0.7 s |
| whisper-base.en | 81 MB | 0.058 | **0.27 s** | **0.58 s** | 0.6 s |
| whisper-small.en | 247 MB | 0.060 | 1.31 s | 2.66 s | 1.2 s |
| distil-small.en | 247 MB | 0.104 | 0.72 s | 1.10 s | 0.7 s |
| whisper-medium.en | ~1 GB | 0.039 | 3.17 s | 5.91 s | 2.6 s |

### On CPU (faster-whisper / CTranslate2 int8 — no NPU backend)

| Model | AVG WER | 6 s clip | 30 s clip | Load |
|---|---|---|---|---|
| small.en | 0.062 | 0.8–4.2 s | 3.66 s | 40 s |
| medium.en | 0.054 | 1.8–9.8 s | 6.73 s | — |

All NPU options transcribe **faster than real-time**. Long audio (>30 s) is chunked
automatically (sliding window) — verified on a 62.5 s concatenation.

### The 30-second encoder floor (why short clips cost the same as long ones)

The static pipeline always encodes a full 30 s window, so latency = **fixed encoder cost**

+ tiny per-word decode. Sweep on distil-medium: 1 s→1.53 s, 6 s→1.46 s, 20 s→1.66 s,
29 s→1.72 s, 62 s (2 windows)→4.81 s. whisper-base's floor is only 0.19 s (6-layer
encoder vs 24). For short dictation clips the encoder floor *is* the latency.

### Published LibriSpeech test-clean WER (for calibration — the 7-clip set is too small)

base ~6 %, small ~3 %, distil-small ~3 %, medium ~2.4 %, distil-medium ~2 %, Parakeet ~1.4 %.

**Decision (2026-08-04, with Jim): distil-medium.en is the primary dictation ASR.** It is
the only NPU option with Parakeet-level quality (~2 % vs ~1.4 %); the 1.5 s floor is a
fine dictation wait. whisper-base.en remains exported as a faster fallback (0.27 s, ~6 %).
Everything else measured worse: whisper-small = distil-medium's latency with worse quality;
distil-small dropped phrases; whisper-medium's 24-layer decoder is 2–3× slower than
base's quality tier for no gain; faster-whisper on CPU was the slowest of all.

+ Silent audio makes Whisper-family models hallucinate short text (e.g. `"You're not going
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

## Integration status (DONE 2026-08-04)

The winner is wired into the production dictation pipeline:

+ **`asr_server.py`** (this dir): FastAPI OpenAI-compatible `/v1/audio/transcriptions`
  over `WhisperPipeline(..., "NPU", static)`; decodes any audio via ffmpeg; serialized
  `generate`; `/health` for llama-swap's `checkEndpoint`. Requires `LD_LIBRARY_PATH`
  (set via the llama-swap entry `env`).
+ **config.yaml**: new `whisper-npu-asr` entry (distil-medium.en-npu) as the `dictation`
  group's ASR member alongside `qwen-clean-2b` (cleanup unchanged). `parakeet-asr` kept
  as an unlisted GPU fallback (`model=parakeet-asr`; flip back with `BARUCH_ASR_MODEL`).
+ **baruch-server**: `ASR_MODEL = "whisper-npu-asr"` (override: `BARUCH_ASR_MODEL`),
  `DICTATION_MODELS` updated.
+ **warmup-stt.sh**: warms `whisper-npu-asr` + `qwen-clean-2b` on boot.
+ Verified live: `/v1/dictate` end-to-end (NPU ASR → qwen cleanup → British pass) in
  ~1.6 s; both group members stay resident (no eviction); ASR uses **no GPU** — dictation
  ASR now works even while 3-GPU --fit models hold GPU 0.

Performance (measured through llama-swap): cold first call ~8 s (server spawn + pipeline
load), warm ~1.5 s/clip.

### Pitfall hit during deployment

The edit tool's YAML auto-formatter flattened the `groups:` block indentation
(`"dictation":` became a root key) — YAML *valid*, but `groups` was null, so llama-swap
saw no group and the two dictation models evicted each other. Fixed by re-indenting the
block. **Lesson: after any config edit, validate *semantics* (`python -c "import yaml; ..."`),
not just syntax.**
