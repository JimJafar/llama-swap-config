#!/usr/bin/env python3
"""Concurrency test: N simultaneous chat requests (short prompt, long answer), total and per-request speed.

Usage: bench/parallel_bench.py MODEL --base URL [--users 1,2,4] [--max-tokens 1024] [--out FILE]

Each user sends one request with its own nonce and a different topic, all at once. Reported per run: total
generated tokens / wall time (the aggregate), and each request's own decode speed from the server's timings.
"""
import argparse
import json
import pathlib
import secrets
import threading
import time
import urllib.request

TOPICS = ["how a city's water supply works", "how a jet engine works", "how bread rises",
          "how a bicycle gear system works", "how tides form", "how a refrigerator keeps food cold",
          "how vaccines train the immune system", "how a bill becomes law in the UK"]


def one(base, model, topic, max_tokens, out):
    body = {"model": model, "max_tokens": max_tokens, "seed": 42,
            "messages": [{"role": "user", "content": f"[run {secrets.token_hex(6)}]\nWrite a detailed essay of "
                                                     f"at least 2,000 words explaining {topic}. Use headed sections."}]}
    req = urllib.request.Request(f"{base}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    t = d.get("timings") or {}
    out.append({"tokens": d["usage"]["completion_tokens"], "wall_s": round(time.perf_counter() - t0, 2),
                "decode_tps": round(t["predicted_per_second"], 1) if t.get("predicted_per_second") else None})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--base", default="http://127.0.0.1:8033")
    ap.add_argument("--users", default="1,2,4")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "results" / "parallel.jsonl"))
    a = ap.parse_args()
    one(a.base, a.model, "how a kettle works", 16, [])      # warm-up / load
    for n in [int(x) for x in a.users.split(",")]:
        res = []
        th = [threading.Thread(target=one, args=(a.base, a.model, TOPICS[i % len(TOPICS)], a.max_tokens, res))
              for i in range(n)]
        t0 = time.perf_counter()
        for t in th:
            t.start()
        for t in th:
            t.join()
        wall = time.perf_counter() - t0
        rec = {"model": a.model, "label": a.label, "date": time.strftime("%Y-%m-%d %H:%M"), "users": n,
               "total_tokens": sum(r["tokens"] for r in res), "wall_s": round(wall, 1),
               "aggregate_tps": round(sum(r["tokens"] for r in res) / wall, 1),
               "per_request_decode_tps": [r["decode_tps"] for r in res]}
        print(json.dumps(rec), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
