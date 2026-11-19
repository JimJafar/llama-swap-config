# Usage: score.py refs.tsv results.jsonl  -> WER (Whisper English normaliser), timing summary
import json, sys, wave, os, statistics
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
snap = "/home/jim/.cache/huggingface/hub/models--openai--whisper-base.en/snapshots/911407f4214e0e1d82085af863093ec0b66f9cd6"
norm = EnglishTextNormalizer(json.load(open(os.path.join(snap, "normalizer.json"))))
refs = dict(l.rstrip("\n").split("\t", 1) for l in open(sys.argv[1]))
def ed(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]
errs = words = 0; proc = []; audio = 0.0; lat = []
for line in open(sys.argv[2]):
    if not line.startswith('{"file"'): continue
    r = json.loads(line); uid = os.path.basename(r["file"])[:-4]
    ref = norm(refs[uid]).split(); hyp = norm(r.get("text", "")).split()
    errs += ed(ref, hyp); words += len(ref)
    w = wave.open(r["file"]); dur = w.getnframes() / w.getframerate(); audio += dur
    ms = sum(v for k, v in r.items() if k.endswith("_ms")); proc.append(ms); lat.append(ms)
n = len(proc)
print(json.dumps({"files": n, "wer_pct": round(100 * errs / words, 2), "audio_s": round(audio, 1),
                  "compute_s": round(sum(proc) / 1000, 2), "rtfx": round(audio / (sum(proc) / 1000), 1),
                  "median_ms": round(statistics.median(lat)), "p90_ms": round(sorted(lat)[int(0.9 * n) - 1])}))
