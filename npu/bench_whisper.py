import sys, time, json, os, wave, re
import numpy as np
import openvino_genai as ov_genai

def load_wav(path):
    with wave.open(path) as w:
        rate, ch = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return data, rate, ch

def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9' ]", " ", s)
    return s.split()

def wer(ref, hyp):
    r = norm(ref)
    h = norm(hyp)
    dp = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): dp[i][0] = i
    for j in range(len(h)+1): dp[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1] + (0 if r[i-1]==h[j-1] else 1))
    return dp[len(r)][len(h)] / max(1, len(r))

def main(model_dir):
    refs = json.load(open("/tmp/asr-bench/refs.json"))
    refs.append({"file": "/tmp/jfk.wav", "ref": "And so my fellow Americans ask not what your country can do for you ask what you can do for your country.", "dur": 11.0})
    t0 = time.time()
    config = {"NPU_USE_NPUW": "YES", "NPUW_DEVICES": "CPU", "NPUW_ONLINE_PIPELINE": "NONE", "STATIC_PIPELINE": True}
    pipe = ov_genai.WhisperPipeline(model_dir, device="NPU", **config)
    print(f"[{os.path.basename(model_dir)}] pipeline load+compile: {time.time()-t0:.1f}s", flush=True)
    tot = 0.0
    for item in refs:
        audio, rate, ch = load_wav(item["file"])
        if rate != 16000 or ch != 1:
            print(f"  SKIP {item['file']} (rate {rate} ch {ch})")
            continue
        t0 = time.time()
        out = pipe.generate(audio, max_new_tokens=100)
        dt = time.time() - t0
        text = out.strip() if isinstance(out, str) else str(out)
        w = wer(item["ref"], text)
        tot += w
        print(f"  {os.path.basename(item['file']):>9} ({item['dur']:>5.1f}s) -> {dt:6.2f}s  WER {w:.3f}", flush=True)
        print(f"    ref: {item['ref'][:90]}")
        print(f"    hyp: {text[:90]}")
    print(f"[{os.path.basename(model_dir)}] AVG WER: {tot/len(refs):.3f}")

if __name__ == "__main__":
    main(sys.argv[1])
