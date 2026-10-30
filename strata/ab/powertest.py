#!/usr/bin/env python3
"""Strata speed under different GPU power limits, cold and warm.
usage: powertest.py LABEL 3090_WATTS 5070TI_WATTS [--reps N] [--model NAME]
--model: another llama-swap model on the same two cards (default: Strata). Strata is unloaded
first (it holds both cards) and reloaded at the end.

Sets the limits through LACT (`lact cli`, wheel group, no sudo), restarts Strata (cold:
empty expert caches), then runs each case once cold and N-1 more times warm:
  long:  ~64K-token prompt, then ~2,048 generated tokens
  short: a short prompt, then ~2,048 generated tokens
GPU power and SM clock are sampled every 0.5 s during each request (nvidia-smi).
Restore the production limits afterwards: powertest.py --restore.
"""
import argparse, json, statistics, subprocess, threading, time, urllib.request, uuid, pathlib
import bench  # same directory: chat(), wait_idle(), HEAD, MODEL, OUT

GPUS = {"3090": "10DE:2204-10DE:147D-0000:02:00.0", "5070ti": "10DE:2C05-1569:F332-0000:03:00.0"}
SMI_INDEX = {"3090": 1, "5070ti": 2}           # nvidia-smi indices on Marvin today
PRODUCTION = {"3090": 280, "5070ti": 260}
MAX_C = 85                                      # stop if either card's core passes this

LONG_PROMPT = ("Above is part of a C++/CUDA inference engine's source and docs. Write a long, detailed "
               "technical review of at least 1,500 words: walk through the main components you saw, how "
               "they fit together, the riskiest code with file names, and a concrete fix for each risk.")
SHORT_PROMPT = ("Write a detailed 1,500-word short story about a lighthouse keeper who discovers the light "
                "is signalling to something under the sea. Rich description, dialogue, a twist ending.")
LONG_CHARS = 211000                             # ~64K tokens at ~3.3 chars/token


def set_limits(w3090, w5070):
    for name, w in (("3090", w3090), ("5070ti", w5070)):
        subprocess.run(["lact", "cli", "-g", GPUS[name], "power-limit", "set", str(w)], check=True,
                       capture_output=True)
    time.sleep(6)                               # LACT re-applies on a 5 s timer
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,power.limit", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {l.split(",")[0].strip(): float(l.split(",")[1]) for l in out.strip().splitlines()}


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.rows, self.stop = [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,power.draw,clocks.sm,temperature.gpu,fan.speed",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
            snap = {}
            for l in out.strip().splitlines():
                i, p, c, t, f = [x.strip() for x in l.split(",")]
                snap[int(i)] = (float(p), float(c), float(t), float(f) if f.replace(".", "").isdigit() else 0.0)
            self.rows.append(snap)
            self.stop.wait(0.5)

    def summary(self):
        s = {}
        for name, i in SMI_INDEX.items():
            p = [r[i][0] for r in self.rows if i in r]
            c = [r[i][1] for r in self.rows if i in r]
            t = [r[i][2] for r in self.rows if i in r]
            f = [r[i][3] for r in self.rows if i in r]
            if p:
                s[name] = {"w_avg": round(statistics.mean(p)), "w_max": round(max(p)),
                           "mhz_avg": round(statistics.mean(c)), "c_max": round(max(t)), "fan_max": round(max(f))}
        return s


STRATA = bench.MODEL


def unload(model):
    urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:8033/api/models/unload/{model}", method="POST"), timeout=120).read()


def restart_strata():
    if bench.MODEL != STRATA:
        unload(STRATA)
    unload(bench.MODEL)
    t = time.time()
    bench.chat([{"role": "user", "content": "Reply with only the word OK."}], 2)
    return round(time.time() - t, 1)


def one(label, phase, case):
    nonce = uuid.uuid4().hex
    if case == "long":
        msg = f"[{nonce}]\n" + bench.HEAD[:LONG_CHARS] + "\n\n" + LONG_PROMPT
    else:
        msg = f"[{nonce}] " + SHORT_PROMPT
    if bench.MODEL == STRATA:
        bench.wait_idle()                       # Strata's /metrics; llama.cpp has none
    smp = Sampler()
    smp.start()
    d = bench.chat([{"role": "user", "content": msg}], 2048)
    smp.stop.set()
    smp.join()
    t = d.get("timings") or {}
    row = {"label": label, "phase": phase, "case": case, "prompt_n": t.get("prompt_n"),
           "prefill_tps": t.get("prompt_per_second"), "gen_n": t.get("predicted_n"),
           "decode_tps": t.get("predicted_per_second"), "draft_n": t.get("draft_n"),
           "draft_acc": t.get("draft_n_accepted"), "wall": round(d["_wall"], 1), "power": smp.summary()}
    print(json.dumps(row), flush=True)
    hot = {n: v["c_max"] for n, v in row["power"].items() if v["c_max"] > MAX_C}
    if hot:
        print(f"TOO HOT {hot}: restoring the production limits and stopping", flush=True)
        set_limits(PRODUCTION["3090"], PRODUCTION["5070ti"])
        raise SystemExit(3)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", nargs="?")
    ap.add_argument("w3090", nargs="?", type=int)
    ap.add_argument("w5070", nargs="?", type=int)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--model", default=STRATA)
    a = ap.parse_args()
    bench.MODEL = a.model
    if a.restore:
        print("restored:", set_limits(PRODUCTION["3090"], PRODUCTION["5070ti"]))
        return
    print("limits:", set_limits(a.w3090, a.w5070), flush=True)
    print("load_s:", restart_strata(), flush=True)
    rows = []
    for rep in range(a.reps):
        phase = "cold" if rep == 0 else "warm"
        for case in ("long", "short"):
            rows.append(one(a.label, phase, case))
    bench.OUT.mkdir(exist_ok=True)
    f = bench.OUT / f"power-{a.label}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("->", f)
    if bench.MODEL != STRATA:
        unload(bench.MODEL)
        bench.MODEL = STRATA                    # bring Strata back
        bench.chat([{"role": "user", "content": "Reply with only the word OK."}], 2)


if __name__ == "__main__":
    main()
