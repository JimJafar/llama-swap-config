#!/usr/bin/env python3
"""Sync ~/.pi/agent/models.json from llama-swap's config.yaml.

- Model list = every entry in config.yaml `models:` that is not `unlisted`
  and not in EXCLUDES.
- Model id = the entry's first `aliases` value when it has one, else the config
  key (e.g. Q3.8-27B-IQ4XS -> `subagent`). llama-swap routes either name, but pi
  sends models.json's id as `model`, and the alias is the name meant for clients.
  `alias:` (singular) is NOT a llama-swap field and is ignored by the server, so
  the server honours `aliases:` only.
- contextWindow = taken from the model's `-c N`, `--fit-ctx N` or
  vLLM `--max-model-len N` flag in its cmd (falls back to whatever the
  current models.json says, then to DEFAULT_CONTEXT_WINDOW).
- maxTokens = MAX_TOKENS_RATIO (95%) of contextWindow, recomputed on every run.
  pi defaults this to 16384, which truncates long-thinking models mid-reasoning.
- Curated per-model fields (reasoning, input, thinkingLevelMap, compat,
  thinkingFormat, ...) are looked up under the new id first, then the config key,
  then preserved from the existing models.json for ids it already has;
  brand-new ids get a minimal entry
  ({id, input: ["text"], contextWindow}) — add their reasoning/thinking
  fields by hand once.
- Models removed from config.yaml (or marked unlisted) drop out.
- THINKING_WIRING: per-family pi thinking/effort wiring (reasoning,
  thinkingLevelMap, compat.chatTemplateKwargs with $var mappings) applied to
  every base model whose id matches the family marker. This is what lets pi
  control thinking on/off + effort per request instead of leaving it to the
  server's baked flags.

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

# pi defaults each model's maxTokens (max OUTPUT tokens, sent as the request's
# max_completion_tokens) to 16384, which truncates heavy thinking models such as
# Qwen3.8 mid-reasoning. Set it to this fraction of the model's context window
# instead, so the output cap is effectively removed. llama.cpp clamps n_predict
# to the space actually left after the prompt, so a value this high is safe.
MAX_TOKENS_RATIO = 0.95

# Per-family pi thinking/effort wiring, applied to every base model whose id
# (lowercased) contains the marker. Overrides reasoning / thinkingLevelMap /
# compat on the base entry (so hand-curated "off": null maps get replaced by
# real controls).
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
    # `marker` may be a string OR a list of strings (see the matching loop in
    # main()). Both conventions are listed because config.yaml was renamed to short
    # ids on 2026-09-17 (Qwen3.8-* -> Q3.8-*); a marker that silently stops
    # matching strips pi's thinking controls with NO error at all.
    {
        "marker": ["q3.8", "qwen3.8"],
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
        "marker": ["q3.5", "qwen3.5"],
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
        "marker": ["q3.6", "qwen3.6"],
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


def model_aliases(entry: dict) -> list[str]:
    """Client-facing names for a model: config `aliases` first, then `alias`.

    llama-swap's field is `aliases` (a list). `alias` (singular) is not a
    llama-swap field at all -- the server ignores it silently -- but it is read
    here as a single-name convenience so a stray singular key in config.yaml
    still reaches models.json instead of vanishing from both places.
    """
    raw = entry.get("aliases")
    if raw is None:
        raw = entry.get("alias")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [a for a in raw if isinstance(a, str)]
    return []


def context_window(cmd: str) -> int | None:
    # `cmd` is a YAML block scalar and commonly contains shell comment lines
    # (`# ...`) that mention flags -- e.g. a note like "the 262K profile is
    # -c 262144" sitting ABOVE the real `-c 180000`. Drop comment lines before
    # matching, so only real flags are parsed.
    body = "\n".join(
        line for line in cmd.splitlines() if not line.lstrip().startswith("#")
    )
    try:
        m = (
            re.search(r"(?:^|\s)-c\s+(\d+)", body)          # llama.cpp
            or re.search(r"--fit-ctx\s+(\d+)", body)        # llama.cpp --fit floor
            or re.search(r"--max-model-len\s+(\d+)", body)   # vLLM
            or re.search(r"--max-seq-len\s+(\d+)", body)     # TabbyAPI / exllamav3
        )
        return int(m.group(1)) if m else None
    except (re.error, ValueError):
        return None


def max_tokens(ctx) -> int:
    """Max OUTPUT tokens for models.json = MAX_TOKENS_RATIO of the context window.

    pi defaults maxTokens to 16384, which truncates long-thinking models; a value
    near the context window effectively removes the cap (llama.cpp clamps n_predict
    to the space left after the prompt). Falls back to the default window when ctx
    is missing or not a usable integer (e.g. a hand-edited models.json).
    """
    try:
        return int(int(ctx) * MAX_TOKENS_RATIO)
    except (TypeError, ValueError):
        return int(DEFAULT_CONTEXT_WINDOW * MAX_TOKENS_RATIO)


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
    ctx_unparsed = []
    renames = []
    for name, entry in cfg["models"].items():
        if entry.get("unlisted") or name in EXCLUDES:
            continue
        aliases = model_aliases(entry)
        model_id = aliases[0] if aliases else name
        if len(aliases) > 1:
            # Only the first alias becomes the pi model id; further aliases stay
            # routable in llama-swap but would be duplicate entries here.
            print(f"note: {name}: extra aliases not exposed to pi:", aliases[1:])
        # Curated fields follow the model, not its name: look up the new id first
        # (that is what models.json used on the previous run), then the config
        # key, so renaming a model to an alias keeps its reasoning/thinking wiring.
        keep: dict[str, object] = dict(cur_models.get(model_id) or cur_models.get(name)
                                       or {"input": ["text"]})
        if model_id != name and name in cur_models:
            renames.append((name, model_id))
        keep["id"] = model_id
        parsed = context_window(entry.get("cmd", ""))
        if parsed is None:
            ctx_unparsed.append(name)
        ctx = parsed or keep.get("contextWindow") or DEFAULT_CONTEXT_WINDOW
        keep["contextWindow"] = ctx
        keep["maxTokens"] = max_tokens(ctx)
        new_models.append(keep)

    # Apply per-family thinking/effort wiring to the base entries. `marker` is a
    # substring of the lowercased model id, and may be a list so one family can
    # carry several naming conventions at once.
    for m in new_models:
        for spec in THINKING_WIRING:
            markers = spec["marker"]
            if isinstance(markers, str):
                markers = [markers]
            if any(mk in m["id"].lower() for mk in markers):
                m.update(spec["fields"])
                break

    out = {
        "providers": {
            "marvin": {**PROVIDER, "models": new_models}
        }
    }

    old_ids = set(cur_models)
    new_ids = {m["id"] for m in new_models}
    print("renamed: ", [f"{a} -> {b}" for a, b in renames] or "nothing")
    print("removing:", sorted(old_ids - new_ids) or "nothing")
    print("adding:  ", sorted(new_ids - old_ids) or "nothing")
    changed_ctx = [m["id"] for m in new_models
                   if m["id"] in old_ids and m["contextWindow"] != cur_models[m["id"]].get("contextWindow")]
    print("ctxWindow changes:", changed_ctx or "none")
    if ctx_unparsed:
        # Silence here is dangerous: an unrecognised flag means the context silently falls
        # back to the EXISTING models.json value (so a changed cmd never propagates) or to
        # DEFAULT_CONTEXT_WINDOW. That is how the EXL3 entry sat at 32768 for hours while
        # its cmd said --max-seq-len 262144. Expected for non-chat backends (TTS/ASR).
        print("WARNING: no context flag recognised in:", sorted(ctx_unparsed),
              f"-> falling back to existing value / {DEFAULT_CONTEXT_WINDOW}")
    changed_max = [m["id"] for m in new_models
                   if m["id"] in old_ids and m["maxTokens"] != cur_models[m["id"]].get("maxTokens")]
    print("maxTokens changes:", changed_max or "none")

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
