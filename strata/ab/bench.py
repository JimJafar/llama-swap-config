#!/usr/bin/env python3
"""A/B benchmark for Strata variants, run through llama-swap's Strata slot.

Each case: unique nonce up front (no prefix reuse), temperature 0, timings from
Strata's llama.cpp-style `timings`. Waits for Strata to be idle before each request
so other clients' work does not overlap ours (theirs queue behind ours instead).

usage: bench.py LABEL [--reps N] [--quick] [--long]
"""
import gzip, json, os, sys, time, uuid, glob, statistics, argparse, urllib.request, pathlib

URL = "http://127.0.0.1:8033/v1/chat/completions"
MODEL = os.environ.get("STRATA_MODEL", "Strata-IQ3XXS")
OUT = pathlib.Path(__file__).resolve().parent / "results"

# The prompts: the start (prefill cases) and end (agent case) of Strata's src, docs and
# serve files at commit 1735d64 (0.1.40, the checkout the 2026-10-11 baseline read),
# frozen in corpus.json.gz so results stay comparable as Strata's own source changes.
_C = json.load(gzip.open(pathlib.Path(__file__).resolve().parent / "corpus.json.gz", "rt"))
HEAD, TAIL = _C["head"], _C["tail"]

def strata_port():
    r = json.load(urllib.request.urlopen("http://127.0.0.1:8033/running", timeout=10))
    for m in r["running"]:
        if m["model"] == MODEL and m["state"] == "ready":
            return m["proxy"].rsplit(":", 1)[1]
    return None

def wait_idle(timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = strata_port()
        if p:
            try:
                live = json.load(urllib.request.urlopen(f"http://127.0.0.1:{p}/metrics", timeout=10))["live"]
                if live.get("state") == "idle" and not live.get("queued"):
                    return
            except Exception:
                pass
        time.sleep(1)

def chat(messages, max_tokens, **extra):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "reasoning_effort": "none", "stream": False, **extra}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"content-type": "application/json"})
    t = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    d["_wall"] = time.time() - t
    return d

# chars per token on this corpus is ~3.3; sizes are approximate, the timings carry the real count
PREFILL = [("p1k", 3300), ("p4k", 13200), ("p16k", 52800), ("p32k", 105600)]
LONG = [("p100k", 330000)]
DECODE = [
    ("code", "Write a complete, well-commented Python implementation of a thread-safe LRU cache with TTL expiry, "
             "a small CLI demo, and pytest tests covering eviction, expiry and concurrency."),
    ("prose", "Write a detailed 900-word short story about a lighthouse keeper who discovers the light is "
              "signalling to something under the sea. Rich description, dialogue, a twist ending."),
    ("explain", "Explain step by step how a modern CPU executes an instruction stream: fetch, decode, rename, "
                "out-of-order issue, branch prediction, caches and memory ordering. Be thorough and concrete."),
]
AGENT = ("agent20k", 66000, "Above is part of a C++/CUDA inference engine's source and docs. Write a concise review: "
         "list the five riskiest pieces of code you saw with file names and why, then propose a fix for each.")

def run(label, reps, quick, long_, only=None):
    OUT.mkdir(exist_ok=True)
    rows = []
    cases = []
    for name, n in PREFILL + (LONG if long_ else []):
        cases.append(("prefill", name, n, None, 32))
    for name, prompt in DECODE:
        cases.append(("decode", name, 0, prompt, 1024))
    cases.append(("agent", AGENT[0], AGENT[1], AGENT[2], 700))
    if quick:
        cases = [c for c in cases if c[1] in ("p4k", "p16k", "code", "prose")]
    if only:
        cases = [c for c in cases if c[1] in only]
    for rep in range(reps):
        for kind, name, n, prompt, mt in cases:
            nonce = uuid.uuid4().hex
            if kind == "prefill":
                msg = f"[{nonce}]\n" + HEAD[:n] + "\n\nReply with only the word OK."
            elif kind == "agent":
                msg = f"[{nonce}]\n" + TAIL[-n:] + "\n\n" + prompt
            else:
                msg = f"[{nonce}] " + prompt
            wait_idle()
            d = chat([{"role": "user", "content": msg}], mt)
            t = d.get("timings") or {}
            row = {"label": label, "rep": rep, "kind": kind, "case": name,
                   "prompt_n": t.get("prompt_n"), "prefill_tps": t.get("prompt_per_second"),
                   "gen_n": t.get("predicted_n"), "decode_tps": t.get("predicted_per_second"),
                   "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"), "wall": round(d["_wall"], 2),
                   "text": (d["choices"][0]["message"].get("content") or "")[:200]}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "text"}), flush=True)
    f = OUT / f"{label}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("\nSUMMARY", label, "->", f)
    by = {}
    for r in rows:
        by.setdefault(r["case"], []).append(r)
    for c, rs in by.items():
        p = [r["prefill_tps"] for r in rs if r["prefill_tps"]]
        dd = [r["decode_tps"] for r in rs if r["decode_tps"]]
        print(f"  {c:9s} prompt_n={rs[0]['prompt_n']:>7}  prefill={statistics.median(p) if p else 0:8.1f}  "
              f"decode={statistics.median(dd) if dd else 0:7.1f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--cases", default="")
    a = ap.parse_args()
    run(a.label, a.reps, a.quick, a.long, [c for c in a.cases.split(",") if c])
