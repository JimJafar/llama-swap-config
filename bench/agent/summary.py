#!/usr/bin/env python3
"""Compare conditions in bench/results/agent.jsonl: per task, the mean of each metric, and the change against
the first label given (the baseline).

Usage: bench/agent/summary.py BASELINE_LABEL [OTHER_LABEL ...]
"""
import json
import pathlib
import statistics
import sys

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results" / "agent.jsonl"
METRICS = [("wall_s", "wall min", 1 / 60), ("output_tokens", "output tok", 1), ("thinking_chars", "thinking chars", 1),
           ("calls", "calls", 1), ("max_context", "max context", 1)]


def main():
    labels = sys.argv[1:]
    rows = [json.loads(line) for line in RESULTS.open()]
    tasks = sorted({r["task"] for r in rows if r["label"] in labels})
    for task in tasks:
        print(f"\n## {task}")
        print(f"| condition | runs | tests | " + " | ".join(name for _, name, _ in METRICS) + " |")
        print("|---|---:|---:|" + "---:|" * len(METRICS))
        base = None
        for label in labels:
            rs = [r for r in rows if r["label"] == label and r["task"] == task]
            if not rs:
                continue
            tests = f"{sum(r['tests_passed'] for r in rs)}/{sum(r['tests_total'] for r in rs)}"
            means = {k: statistics.mean(r.get(k) or 0 for r in rs) * f for k, _, f in METRICS}
            ranges = {k: (min(r.get(k) or 0 for r in rs) * f, max(r.get(k) or 0 for r in rs) * f) for k, _, f in METRICS}
            if base is None:
                base = means
            cells = []
            for k, _, f in METRICS:
                v = means[k]
                txt = f"{v:,.1f}" if f != 1 else f"{v:,.0f}"
                if base is not means and base[k]:
                    txt += f" ({(v / base[k] - 1) * 100:+.0f}%)"
                lo, hi = ranges[k]
                if len(rs) > 1:
                    txt += f" [{lo:,.1f}–{hi:,.1f}]" if f != 1 else f" [{lo:,.0f}–{hi:,.0f}]"
                cells.append(txt)
            print(f"| {label} | {len(rs)} | {tests} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
