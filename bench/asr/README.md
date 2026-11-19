# CPU-only speech-to-text benchmark (2026-11-19)

Which STT model to run on Marvin's CPU, now that whisper-npu-asr has no NPU. Results:
[`../results/asr-cpu-2026-11-19.jsonl`](../results/asr-cpu-2026-11-19.jsonl).

- Engine: transcribe.cpp (github.com/handy-computer/transcribe.cpp, `adfc5021`) in `~/lab/transcribe.cpp`,
  a CPU-only build: `cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF`.
- Models: the Q8_0 GGUFs of `handy-computer/<model>-gguf` in `~/lab/transcribe.cpp/models/<model>/`
  (SHA-256 checked against the Hugging Face LFS hashes).
- Audio: 80 LibriSpeech test-other utterances, 8 from each of 10 offsets (0, 290, ... 2610) of the
  `openslr/librispeech_asr` `other`/`test` split via datasets-server; ids and transcripts in `refs.tsv`.
  Stored as 16 kHz WAV in `~/lab/asr-bench/audio/<id>.wav` (rebuildable from the ids).
- Score: `score.py`, WER after Whisper's EnglishTextNormalizer (the Open ASR Leaderboard's); time is
  transcribe.cpp's own mel + encode + decode per file, model load excluded. The 29.5 s clip is the
  first six utterances concatenated.

1367 reference words: 1 point of WER is about 14 words, so gaps under ~1 point are not decisive.
