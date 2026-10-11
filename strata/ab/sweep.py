#!/usr/bin/env python3
"""A/B points through llama-swap's Strata slot: each is an engine config, optionally with args changed.

usage: sweep.py BASE.json NAME:"--arg v ENV:VAR=v ..." [NAME:"..."] ... [--cases a,b] [--reps N]

For each point: write ab/strata-sw-NAME.json (BASE with those args replaced or added,
and ENV: entries set in its env), point ../ab.env at it, restart the model once it is
idle, and run bench.py with label sw-NAME. "NAME:" with no args runs BASE as it is.
At the end ab.env is removed and the model restarted, so production is back on the
config llama-swap names.

The model is $STRATA_MODEL (default Strata-IQ3S).
"""
import json, os, sys, subprocess, time, urllib.request, shlex, shutil, pathlib, argparse

AB = pathlib.Path(__file__).resolve().parent
ABENV = AB.parent / "ab.env"
MODEL = os.environ.get("STRATA_MODEL", "Strata-IQ3S")

def running_port():
    r = json.load(urllib.request.urlopen("http://127.0.0.1:8033/running", timeout=10))
    for m in r["running"]:
        if m["model"] == MODEL and m["state"] == "ready":
            return m["proxy"].rsplit(":", 1)[1]

def wait_idle():
    while True:
        p = running_port()
        if not p:
            return
        try:
            live = json.load(urllib.request.urlopen(f"http://127.0.0.1:{p}/metrics", timeout=10))["live"]
            if live.get("state") == "idle" and not live.get("queued"):
                return
        except Exception:
            pass
        time.sleep(2)

def restart():
    wait_idle()
    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:8033/api/models/unload/{MODEL}",
                                                  method="POST"), timeout=60).read()
    while subprocess.run(["pgrep", "-x", "strata"], capture_output=True).returncode == 0:
        time.sleep(1)
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 8, "reasoning_effort": "none"}).encode()
    d = json.load(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8033/v1/chat/completions", body, {"content-type": "application/json"}), timeout=900))
    assert d.get("choices"), d

def point_config(base, args):
    c = json.load(open(base))
    extra = shlex.split(args)
    a = c["args"]
    i = 0
    while i < len(extra):   # an arg given here replaces the base's value for it
        k = extra[i]
        has_val = i + 1 < len(extra) and not extra[i + 1].startswith("--")
        if k.startswith("ENV:"):
            kk, vv = k[4:].split("=", 1)
            c.setdefault("env", {})[kk] = vv
            i += 1
            continue
        if k in a:
            j = a.index(k)
            if has_val and j + 1 < len(a) and not a[j + 1].startswith("--"):
                a[j + 1] = extra[i + 1]
        else:
            a.append(k)
            if has_val:
                a.append(extra[i + 1])
        i += 2 if has_val else 1
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("points", nargs="+")
    ap.add_argument("--cases", default="")
    ap.add_argument("--reps", type=int, default=2)
    o = ap.parse_args()
    base_shared = pathlib.Path(o.base).with_suffix("").as_posix() + ".shared-settings.json"
    try:
        for spec in o.points:
            name, args = spec.split(":", 1)
            cfg = AB / f"strata-sw-{name}.json"
            json.dump(point_config(o.base, args), open(cfg, "w"), indent=1)
            if os.path.exists(base_shared):
                shutil.copy(base_shared, AB / f"strata-sw-{name}.shared-settings.json")
            ABENV.write_text(f"# sweep point {name}: {args}\nCONFIG={cfg}\n")
            print(f"=== {name}: {args or '(base)'}", flush=True)
            restart()
            cmd = [sys.executable, str(AB / "bench.py"), f"sw-{name}", "--reps", str(o.reps)]
            if o.cases:
                cmd += ["--cases", o.cases]
            subprocess.run(cmd, check=False)
    finally:
        ABENV.unlink(missing_ok=True)
        print("=== back to production", flush=True)
        restart()

main()
