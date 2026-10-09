#!/usr/bin/env python3
"""Speed test for llama-swap models: load time, short prompt + long generation,
long prompt (~30K tokens) + short generation.

Usage: bench/llm_bench.py MODEL [MODEL...] [--base URL] [--out results.jsonl]

Each prompt starts with a random nonce, so no prefix cache is reused. Speeds come
from llama-server's own `timings` when the backend returns them, otherwise from the
client clock (prefill = prompt tokens / time to first token, decode = tokens after the
first / time after the first). Sampling is left to each model's config; seed 42.
"""
import argparse
import json
import pathlib
import secrets
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
LONG_DOC = (HERE.parent / "test-30K.md").read_text()

TESTS = {
    "short-prompt/long-gen": dict(
        prompt=("Write a detailed technical essay of at least 3,000 words on how a large city's "
                "water supply works, from reservoir to tap: sources, treatment stages, pumping, "
                "storage, distribution networks, pressure management, leak detection and "
                "maintenance. Use headed sections and full paragraphs."),
        max_tokens=2048),
    "long-prompt": dict(
        prompt=LONG_DOC + "\n\nIn about 300 words, summarise the main events of this chronicle.",
        max_tokens=512),
}


def chat(base, model, prompt, max_tokens, timeout=1800):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": f"[run {secrets.token_hex(6)}]\n{prompt}"}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": 42,
    }
    req = urllib.request.Request(f"{base}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = None
    n_chunks = 0
    usage = timings = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            if ev.get("timings"):
                timings = ev["timings"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    n_chunks += 1
                    if t_first is None:
                        t_first = time.perf_counter()
    t_end = time.perf_counter()
    usage = usage or {}
    p_tok = usage.get("prompt_tokens")
    c_tok = usage.get("completion_tokens") or n_chunks
    res = dict(wall_s=round(t_end - t0, 2), prompt_tokens=p_tok, completion_tokens=c_tok,
               ttft_s=round(t_first - t0, 2) if t_first else None)
    if timings and timings.get("predicted_per_second"):
        res.update(prefill_tps=round(timings["prompt_per_second"], 1),
                   decode_tps=round(timings["predicted_per_second"], 1),
                   prompt_eval_tokens=timings.get("prompt_n"), source="server timings")
        if "draft_n" in timings and timings.get("draft_n"):
            res["draft_accept"] = round(timings["draft_n_accepted"] / timings["draft_n"], 3)
    elif t_first:
        res.update(prefill_tps=round(p_tok / (t_first - t0), 1) if p_tok else None,
                   decode_tps=round((c_tok - 1) / (t_end - t_first), 1) if c_tok > 1 else None,
                   source="client clock")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--base", default="http://127.0.0.1:8033")
    ap.add_argument("--out", default=str(HERE / "results" / "llm.jsonl"))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for model in a.models:
        rec = {"model": model, "date": time.strftime("%Y-%m-%d %H:%M")}
        t0 = time.perf_counter()
        try:
            chat(a.base, model, "Reply with OK.", 8)
            rec["load_s"] = round(time.perf_counter() - t0, 1)
            for name, t in TESTS.items():
                rec[name] = chat(a.base, model, t["prompt"], t["max_tokens"])
        except Exception as e:  # keep going with the next model
            rec["error"] = repr(e)
        print(json.dumps(rec), flush=True)
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
