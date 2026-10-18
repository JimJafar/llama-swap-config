#!/usr/bin/env python3
"""Agent benchmark: one coding task, run headless through pi against a llama-swap model, graded by hidden tests.

Usage: bench/agent/run.py TASK LEVEL LABEL [--model marvin/Strata-IQ3S] [--append FILE] [--timeout 2700]

TASK is a folder in tasks/ (prompt.md, workspace/, hidden/). LEVEL is pi's thinking level (pi's model map turns
it into the request's reasoning_effort). LABEL names the condition in the results.

Each run gets a fresh copy of the workspace under ~/lab/agent-bench/runs/. pi runs with no extensions, skills,
prompt templates, context files or MCP, so only the model, the template and --append differ between conditions.
Afterwards the task's original visible tests and its hidden tests are run against the result (the agent's own
copies of the tests are not trusted), and the timings and token counts are read from pi's session file.
Results are appended to bench/results/agent.jsonl.
"""
import argparse
import datetime
import json
import pathlib
import re
import shutil
import subprocess
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
RUNS = pathlib.Path.home() / "lab" / "agent-bench" / "runs"
RESULTS = HERE.parent / "results" / "agent.jsonl"


def ts(e):
    return datetime.datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")).timestamp()


def session_metrics(session_dir: pathlib.Path) -> dict:
    files = sorted(session_dir.rglob("*.jsonl"))
    if not files:
        return {"error": "no session file"}
    entries = [json.loads(line) for f in files for line in f.open()]
    calls = model_s = tool_s = out_tok = think_chars = args_chars = text_chars = 0
    max_ctx = 0
    tools = {}
    errors = 0
    first = last = None
    last_end = None
    for e in entries:
        if e.get("type") != "message":
            continue
        m = e["message"]
        if m["role"] == "user" and first is None:
            first = ts(e)
        elif m["role"] == "assistant":
            start, end = m["timestamp"] / 1000, ts(e)
            calls += 1
            model_s += end - start
            last = last_end = end
            u = m.get("usage") or {}
            out_tok += u.get("output") or 0
            max_ctx = max(max_ctx, (u.get("input") or 0) + (u.get("cacheRead") or 0) + (u.get("output") or 0))
            for c in m["content"]:
                if c["type"] == "thinking":
                    think_chars += len(c.get("thinking", ""))
                elif c["type"] == "text":
                    text_chars += len(c.get("text", ""))
                elif c["type"] == "toolCall":
                    tools[c["name"]] = tools.get(c["name"], 0) + 1
                    args_chars += len(json.dumps(c.get("arguments", {})))
        elif m["role"] == "toolResult":
            if last_end is not None:
                tool_s += ts(e) - last_end
                last_end = ts(e)
            errors += bool(m.get("isError"))
    return dict(wall_s=round((last or 0) - (first or 0), 1), model_s=round(model_s, 1), tool_s=round(tool_s, 1),
                calls=calls, output_tokens=out_tok, thinking_chars=think_chars, text_chars=text_chars,
                tool_args_chars=args_chars, max_context=max_ctx, tool_calls=tools, tool_errors=errors)


def grade(task: pathlib.Path, work: pathlib.Path) -> dict:
    """Run the task's original visible tests and its hidden tests against the agent's code."""
    g = work.parent / "grade"
    shutil.rmtree(g, ignore_errors=True)
    shutil.copytree(work, g, ignore=shutil.ignore_patterns("__pycache__"))
    # The original visible tests (and the files beside them, such as sample data) and the hidden tests go into one
    # folder of their own, so tests the agent wrote or edited are not part of the grade.
    hidden_dir = g / "_grade"
    hidden_dir.mkdir()
    for f in (task / "workspace").rglob("*"):
        if f.is_file() and ("test" in f.name or f.parent.name == "tests"):
            shutil.copy(f, hidden_dir / f.name)
    for f in (task / "hidden").glob("*.py"):
        shutil.copy(f, hidden_dir / f.name)
    r = subprocess.run(["timeout", "120", "python3", "-m", "unittest", "discover", "-s", "_grade"],
                       cwd=g, capture_output=True, text=True)
    out = r.stderr + r.stdout
    m = re.search(r"Ran (\d+) test", out)
    total = int(m.group(1)) if m else 0
    f = re.search(r"FAILED \((.*)\)", out)
    failed = sum(int(x) for x in re.findall(r"=(\d+)", f.group(1))) if f else 0
    lines = out.strip().splitlines()
    expected = int((task / "expected_tests").read_text())   # a module that fails to import counts as all failed
    return {"tests_passed": total - failed if m else 0, "tests_total": max(total, expected),
            "grade_detail": lines[-1] if lines else str(r.returncode)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("level")
    ap.add_argument("label")
    ap.add_argument("--model", default="marvin/Strata-IQ3S")
    ap.add_argument("--append", help="file appended to pi's system prompt")
    ap.add_argument("--timeout", type=int, default=2700)
    a = ap.parse_args()
    task = HERE / "tasks" / a.task
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run = RUNS / f"{stamp}-{a.label}-{a.task}"
    work = run / "work"
    shutil.copytree(task / "workspace", work)
    prompt = (task / "prompt.md").read_text()
    cmd = ["pi", "-p", "--model", a.model, "--thinking", a.level, "--session-dir", str(run / "session"),
           "-ne", "-ns", "-nc", "-np", "--no-mcp", "--offline"]
    if a.append:
        cmd += ["--append-system-prompt", str(pathlib.Path(a.append).resolve())]
    cmd += ["--", prompt]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=a.timeout)
        status = "ok" if r.returncode == 0 else f"exit {r.returncode}"
        (run / "pi.out").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    except subprocess.TimeoutExpired:
        status = "timeout"
    rec = {"task": a.task, "label": a.label, "level": a.level, "model": a.model, "append": a.append,
           "date": time.strftime("%Y-%m-%d %H:%M"), "status": status, "run_dir": str(run),
           "elapsed_s": round(time.time() - t0, 1)}
    rec.update(session_metrics(run / "session"))
    rec.update(grade(task, work))
    print(json.dumps(rec), flush=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
