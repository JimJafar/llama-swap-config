#!/usr/bin/env python3
"""Inkling speed check through llama-swap: code / prose / story generation, then the
test-30K.md read and the decode after it. Streamed; client clock (decode = tokens after
the first / time after the first; prefill = prompt tokens / TTFT). Random nonce per run.
usage: ink_bench.py MODEL [ROUNDS] [REASONING_EFFORT, default none]"""
import json, secrets, sys, time, urllib.request

BASE = "http://127.0.0.1:8033"
DOC = open("/home/jim/llama-swap/test-30K.md").read()
TESTS = [
    ("code", "Write a complete Python module implementing a thread-safe LRU cache with TTL expiry, "
             "size-based eviction, statistics, and a decorator interface. Include type hints, docstrings "
             "and a full pytest test suite.", 1500),
    ("prose", "Write a detailed technical essay of at least 3,000 words on how a large city's water supply "
              "works, from reservoir to tap. Use headed sections and full paragraphs.", 1500),
    ("story", "Write a long short story (about 2,500 words) about a lighthouse keeper who finds a message "
              "in a bottle addressed to her by name.", 1500),
    ("30K", DOC + "\n\nIn about 300 words, summarise the main events of this chronicle.", 512),
]


def run(model, prompt, max_tokens, effort):
    body = {"model": model, "messages": [{"role": "user", "content": f"[run {secrets.token_hex(6)}]\n{prompt}"}],
            "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True}}
    if effort:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter(); tf = None; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if tf is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                    tf = time.perf_counter()
    t1 = time.perf_counter()
    pt, ct = usage["prompt_tokens"], usage["completion_tokens"]
    return dict(prompt_tokens=pt, completion_tokens=ct, ttft=round(tf - t0, 2),
                prefill_tps=round(pt / (tf - t0)), decode_tps=round((ct - 1) / (t1 - tf), 1))


model = sys.argv[1]
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
effort = sys.argv[3] if len(sys.argv) > 3 else "none"
run(model, "Say hi.", 16, effort)  # load + warm
for name, prompt, mt in TESTS:
    for i in range(rounds if name != "30K" else 1):
        r = run(model, prompt, mt, effort)
        print(json.dumps({"model": model, "test": name, "round": i, "effort": effort, **r}), flush=True)
