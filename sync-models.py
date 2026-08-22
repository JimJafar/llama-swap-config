#!/usr/bin/env python3
"""Sync ~/.pi/agent/models.json from llama-swap's config.yaml.

- Model list = every entry in config.yaml `models:` that is not `unlisted`
  and not in EXCLUDES.
- contextWindow = taken from the model's `-c N` or `--fit-ctx N` flag in
  its cmd (falls back to whatever the current models.json says, then to
  DEFAULT_CONTEXT_WINDOW).
- Curated per-model fields (reasoning, input, thinkingLevelMap, compat,
  thinkingFormat, ...) are preserved from the existing models.json for
  ids it already has; brand-new ids get a minimal entry
  ({id, input: ["text"], contextWindow}) — add their reasoning/thinking
  fields by hand once.
- Models removed from config.yaml (or marked unlisted) drop out.
- For every model whose id contains the INSTRUCT_ID_MARKER, an additional
  "<id>-instruct" variant is emitted with the INSTRUCT_SAMPLING recipe
  (model-card instruct-mode sampling, reasoning off). The variant routes to
  the SAME llama-swap upstream via the model's `aliases` in config.yaml;
  without that alias, llama-swap 404s ("no router for requested model").
- THINKING_WIRING: per-family pi thinking/effort wiring (reasoning,
  thinkingLevelMap, compat.chatTemplateKwargs with $var mappings) applied to
  every base model whose id matches the family marker. This is what lets pi
  control thinking on/off + effort per request instead of leaving it to the
  server's baked flags. The -instruct variants strip all of it (fixed off).

Then copies the result to ~/.pi/agent/models.json.

Usage:  sync-models.py          # write + copy
        sync-models.py --dry-run  # print diff only
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

CONFIG = Path.home() / "llama-swap/config.yaml"
OUT = Path.home() / "llama-swap/models.json"
DEST = Path.home() / ".pi/agent/models.json"

# Models in config.yaml that should NOT appear in models.json.
# unlisted entries are skipped automatically; these are the rest.
EXCLUDES = {
    "whisper-npu-asr",      # dictation (NPU ASR)
    "qwen-clean-2b",        # dictation (prompt cleaner)
    "qwen38-27b-nvfp4",     # vLLM 27B — not exposed to pi (remove to include)
}

DEFAULT_CONTEXT_WINDOW = 32768

# Instruct-mode variants: for any model id containing INSTRUCT_ID_MARKER, also
# emit "<id>-instruct" with this sampling recipe (Qwen3.8 model-card non-thinking
# mode). The llama-swap config.yaml entry must alias the instruct id to the base
# model (`aliases: ["<id>-instruct"]`) or requests to it 404.
#
# chat_template_kwargs.enable_thinking=false is injected here (not via pi's
# thinkingFormat compat) because pi only sends chat_template_kwargs itself for
# models with reasoning: true — its thinkingFormat branches are gated on that.
# With reasoning: false, samplingParams is the only way to force thinking off on
# every request. llama.cpp merges request chat_template_kwargs per-key over the
# server's --chat-template-kwargs and ERASES reasoning_effort when
# enable_thinking is false (tools/server/server-common.cpp), so the base entry's
# server-side "reasoning on / reasoning_effort medium" defaults are overridden.
INSTRUCT_ID_MARKER = "Qwen3.8"
INSTRUCT_ID_SUFFIX = "-instruct"
INSTRUCT_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.80,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    # thinking off on every request (see note above)
    "chat_template_kwargs": {"enable_thinking": False},
}

# Per-family pi thinking/effort wiring, applied to every base model whose id
# (lowercased) contains the marker. Overrides reasoning / thinkingLevelMap /
# compat on the base entry (so hand-curated "off": null maps get replaced by
# real controls). Only the base entries are wired; the -instruct variants are
# generated after and strip reasoning controls entirely (fixed off).
#
# Mechanic: thinkingFormat "chat-template" + chatTemplateKwargs with
# {"$var": "thinking.enabled"} / {"$var": "thinking.effort"} makes pi send
# chat_template_kwargs on every request (the branch is gated on reasoning:
# true, so that flag is required). llama.cpp merges per-request
# chat_template_kwargs per-key over the server's --chat-template-kwargs and
# erases reasoning_effort when enable_thinking is false.
#
# thinkingLevelMap keys are pi levels (off is NOT listed - it is driven by the
# $var, not the map); values are the provider effort string, or null to mark
# the level unsupported (hidden/clamped by pi). Values must match what each
# template accepts - some templates throw on unknown effort values.
THINKING_WIRING = [
    # Qwen3.8: full on/off + effort. Template accepts low/medium/high/xhigh
    # (qwen-3_8-improved-chat-template.jinja validates and raises otherwise).
    {
        "marker": "qwen3.8",
        "fields": {
            "reasoning": True,
            "thinkingLevelMap": {
                "minimal": None, "low": "low", "medium": "medium",
                "high": "high", "xhigh": "xhigh", "max": None,
            },
            "compat": {
                "thinkingFormat": "chat-template",
                "chatTemplateKwargs": {
                    "enable_thinking": {"$var": "thinking.enabled"},
                    "reasoning_effort": {"$var": "thinking.effort"},
                },
            },
        },
    },
    # DeepSeek-V4: full on/off + effort. Embedded template uses `thinking`
    # (default false) and reasoning_effort none/low/high/max.
    {
        "marker": "deepseek",
        "fields": {
            "reasoning": True,
            "thinkingLevelMap": {
                "minimal": None, "low": "low", "medium": None,
                "high": "high", "xhigh": "max", "max": "max",
            },
            "compat": {
                "thinkingFormat": "chat-template",
                "chatTemplateKwargs": {
                    "thinking": {"$var": "thinking.enabled"},
                    "reasoning_effort": {"$var": "thinking.effort"},
                },
            },
        },
    },
    # Qwen3.5 / Qwen3.6 (froggeric fixed template): on/off only - it has
    # enable_thinking but NO reasoning_effort kwarg, so effort levels do
    # nothing server-side. Empty thinkingLevelMap = pi defaults (off allowed).
    {
        "marker": "qwen3.5",
        "fields": {
            "reasoning": True,
            "thinkingLevelMap": {},
            "compat": {
                "thinkingFormat": "chat-template",
                "chatTemplateKwargs": {
                    "enable_thinking": {"$var": "thinking.enabled"},
                },
            },
        },
    },
    {
        "marker": "qwen3.6",
        "fields": {
            "reasoning": True,
            "thinkingLevelMap": {},
            "compat": {
                "thinkingFormat": "chat-template",
                "chatTemplateKwargs": {
                    "enable_thinking": {"$var": "thinking.enabled"},
                },
            },
        },
    },
    # Laguna: on/off only (template: enable_thinking default true, no effort).
    {
        "marker": "laguna",
        "fields": {
            "reasoning": True,
            "thinkingLevelMap": {},
            "compat": {
                "thinkingFormat": "chat-template",
                "chatTemplateKwargs": {
                    "enable_thinking": {"$var": "thinking.enabled"},
                },
            },
        },
    },
]

PROVIDER = {
    "baseUrl": "https://marvin.akita-betelgeuse.ts.net:8033/v1",
    "api": "openai-completions",
    "apiKey": "llama-swap",
}


def context_window(cmd: str) -> int | None:
    try:
        m = re.search(r"(?:^|\s)-c\s+(\d+)", cmd) or re.search(r"--fit-ctx\s+(\d+)", cmd)
        return int(m.group(1)) if m else None
    except (re.error, ValueError):
        return None


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    try:
        cfg = yaml.safe_load(CONFIG.read_text())
        current = json.loads(DEST.read_text())
    except (OSError, yaml.YAMLError, json.JSONDecodeError) as e:
        sys.exit(f"error reading {CONFIG} / {DEST}: {e}")
    if not isinstance(cfg, dict) or not isinstance(cfg.get("models"), dict) or \
            not isinstance(current, dict) or not isinstance(current.get("providers"), dict) or \
            not isinstance(current["providers"].get("marvin"), dict) or \
            not isinstance(current["providers"]["marvin"].get("models"), list):
        sys.exit(f"error: unexpected structure in {CONFIG} or {DEST}")
    cur_models = {m["id"]: m for m in current["providers"]["marvin"]["models"]}

    new_models = []
    for name, entry in cfg["models"].items():
        if entry.get("unlisted") or name in EXCLUDES:
            continue
        keep = dict(cur_models.get(name, {"id": name, "input": ["text"]}))
        keep["id"] = name
        ctx = context_window(entry.get("cmd", "")) or keep.get("contextWindow") or DEFAULT_CONTEXT_WINDOW
        keep["contextWindow"] = ctx
        new_models.append(keep)

    # Apply per-family thinking/effort wiring to the base entries (before the
    # instruct variants are generated, so instructs stay stripped of it).
    for m in new_models:
        for spec in THINKING_WIRING:
            if spec["marker"] in m["id"].lower():
                m.update(spec["fields"])
                break

    # Instruct variants: clone the base entry (input/contextWindow/curated fields),
    # swap the id, and pin the instruct sampling recipe (which forces thinking off
    # via chat_template_kwargs.enable_thinking=false) + reasoning off. Generated
    # fresh every run, so the recipe below is the source of truth (hand-edits to an
    # instruct entry in models.json are overwritten on the next sync). Reasoning
    # controls are stripped so the fixed-off behavior can't be undone by base wiring.
    instruct_models = []
    for m in new_models:
        if INSTRUCT_ID_MARKER in m["id"]:
            inst = dict(m)
            inst["id"] = m["id"] + INSTRUCT_ID_SUFFIX
            inst["name"] = m["id"] + " (instruct)"
            inst["samplingParams"] = dict(INSTRUCT_SAMPLING)
            inst["reasoning"] = False
            inst.pop("compat", None)
            inst.pop("thinkingLevelMap", None)
            instruct_models.append(inst)
    new_models = new_models + instruct_models

    out = {
        "providers": {
            "marvin": {**PROVIDER, "models": new_models}
        }
    }

    old_ids = set(cur_models)
    new_ids = {m["id"] for m in new_models}
    print("removing:", sorted(old_ids - new_ids) or "nothing")
    print("adding:  ", sorted(new_ids - old_ids) or "nothing")
    changed_ctx = [m["id"] for m in new_models
                   if m["id"] in old_ids and m["contextWindow"] != cur_models[m["id"]].get("contextWindow")]
    print("ctxWindow changes:", changed_ctx or "none")

    if dry_run:
        print("(dry run — nothing written)")
        return

    text = json.dumps(out, indent=2) + "\n"
    OUT.write_text(text)
    shutil.copy2(OUT, DEST)
    print(f"wrote {OUT}")
    print(f"copied to {DEST}")
    print("sending to guybrush & ssdnodes")
    src = str(Path.home() / "llama-swap/models.json")
    targets = [
        ("-P42", "ssd-nodes.akita-betelgeuse.ts.net:~/.pi/agent/models.json"),
        (None, "guybrush.akita-betelgeuse.ts.net:~/.pi/agent/models.json"),
    ]
    for opt, dest in targets:
        args = ["scp", src, dest] if opt is None else ["scp", opt, src, dest]
        try:
            subprocess.run(args, check=True)
        except subprocess.CalledProcessError:
            print(f"  WARNING: scp to {dest} failed; local files are written, copy manually")
    print("done")


if __name__ == "__main__":
    main()
