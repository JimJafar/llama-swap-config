#!/usr/bin/env python3
"""Sync ~/.pi/agent/models.json from llama-swap's config.yaml.

- Model list = every entry in config.yaml `models:` that is not `unlisted`
  and not in EXCLUDES.
- Model id = the entry's first `aliases` value when it has one, else the config
  key (e.g. Strata-IQ3S -> `subagent`). llama-swap routes either name, but pi
  sends models.json's id as `model`, and the alias is the name meant for clients.
  An aliased model is ALSO listed under its config key (same fields), so pi shows
  the real model name beside the alias (2026-09-30).
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

Codex gets the same model list in Codex's own format:
~/.codex/model-catalogs/marvin-models.json (see CODEX_CATALOG below). Without it Codex
prints "Model metadata for `subagent` not found. Defaulting to fallback metadata" and runs
the marvin models with a guessed context window, no reasoning levels and no instructions.

Usage:  sync-models.py          # write + copy
        sync-models.py --dry-run  # print diff only
"""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

CONFIG = Path.home() / "llama-swap/config.yaml"
OUT = Path.home() / "llama-swap/models.json"
DEST = Path.home() / ".pi/agent/models.json"
# pi has no per-model default thinking level in models.json; it reads one from
# settings.json `modelThinkingLevels` ("provider/modelId" -> level), which beats the
# global `defaultThinkingLevel`. This script OWNS every "marvin/*" key there: it
# writes one per model whose THINKING_WIRING spec has a `defaultLevel` and removes
# the rest, so set levels in THINKING_WIRING, not by hand. Other providers' keys and
# all other settings are left alone. Local only -- not pushed to guybrush/ssdnodes.
PI_SETTINGS = Path.home() / ".pi/agent/settings.json"
PROVIDER_NAME = "marvin"

# --- Codex -------------------------------------------------------------------
# Codex has no per-provider model list: `model_catalog_json` points at ONE JSON file that
# REPLACES the whole catalog (verified on 0.167: set it in config.toml and the gpt-6 models
# vanish from the picker). So this file is meant to be referenced from the marvin PROFILE
# (~/.codex/marvin.config.toml, `codex -p marvin`), not from the top of config.toml.
# Required per entry: slug, display_name, default_reasoning_level, supported_reasoning_levels,
# shell_type, visibility, supported_in_api, priority, support_verbosity, truncation_policy,
# experimental_supported_tools, and base_instructions (or model_messages.instructions_template).
# Unknown extra keys are ignored; omitted optional keys take the defaults Codex showed for
# them, which are repeated here explicitly where the default is wrong for a local model.
CODEX_HOME = Path.home() / ".codex"
CODEX_CONFIG = CODEX_HOME / "config.toml"
# Codex's new-style profile file: `codex -p marvin` layers this over config.toml. It used to be
# a `[profiles.marvin]` table inside config.toml, and 0.167 REFUSES `-p marvin` while that legacy
# table is still there ("--profile `marvin` cannot be used while ... contains legacy
# `[profiles.marvin]` config"), so ensure_codex_profile() moves it out. Synced to every machine.
CODEX_PROFILE = CODEX_HOME / "marvin.config.toml"
CODEX_CATALOG = CODEX_HOME / "model-catalogs/marvin-models.json"
# How Codex wants the path written: relative to CODEX_HOME, so one file works on all four boxes.
CODEX_CATALOG_REL = "model-catalogs/marvin-models.json"
# base_instructions is required, and Codex's own prompt for gpt-6 is ~19K of GPT/code-mode
# specific text (JS `functions.exec` batching, Apps/Connectors, plugins, `commentary`/`final`
# channels, persistent mode). Hand a 27B model that and you spend a fifth of its window on
# instructions it cannot act on, so this is a distilled version: the parts that change behaviour
# for any model (autonomy, permission, formatting, shell habits) and the tools these models
# actually get (exec_command / write_stdin / view_image / skills). Edit the file, not the script.
CODEX_INSTRUCTIONS = Path(__file__).resolve().parent / "codex/marvin-instructions.md"
# Effort strings Codex's own UI/config understands. A model's levels are the intersection of
# this with what its chat template accepts (the THINKING_WIRING spec), because Codex sends the
# string through verbatim as `reasoning.effort` and templates such as Qwen3.8's RAISE on an
# effort they do not know.
CODEX_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max", "ultra")
# Ids whose model takes images as well as text. Matched as a substring of the id.
CODEX_IMAGE_INPUT_MARKERS = ("vision",)

# Models in config.yaml that should NOT appear in models.json.
# unlisted entries are skipped automatically; these are the rest.
EXCLUDES = {
    "whisper-npu-asr",      # dictation (NPU ASR)
    "qwen-clean-2b",        # dictation (prompt cleaner)
    "qwen38-27b-nvfp4",     # vLLM 27B — not exposed to pi (remove to include)
}

DEFAULT_CONTEXT_WINDOW = 32768

# Models whose context window cannot be parsed out of the cmd, because the cmd
# is a launcher script whose real flags live in a file that script reads. Without
# an entry here they fall back to DEFAULT_CONTEXT_WINDOW and pi truncates early.
CONTEXT_OVERRIDES: dict[str, int] = {}

# Strata: the launcher reads a strata-*.json engine config, and --max-context in its
# `args` is the real limit. Read it at sync time rather than hard-coding it here (a
# hard-coded 65536 went stale when Swift moved to 262144 on 2026-09-30).
STRATA_CONFIGS: dict[str, Path] = {
    # 2026-10-12: Marvin's tuned configs in strata/ (official Strata 0.1.43; the fork is retired)
    # 2026-11-06: none (Strata-IQ3S and its config were removed).
}


# Basalt (2026-11-01): the same idea. Basalt 0.3's v2 config has the context in its model
# entry ("models": [{"max-context": N}]), the v1 one in "engine".
BASALT_CONFIGS: dict[str, Path] = {
    "Basalt-Q4KXL": Path(__file__).resolve().parent / "basalt" / "basalt-q4kxl.json",
}


def strata_context(model_id: str) -> int | None:
    bpath = BASALT_CONFIGS.get(model_id)
    if bpath is not None:
        try:
            c = json.loads(bpath.read_text())
            m = (c.get("models") or [{}])[0]
            return int(m.get("max-context") or c["engine"]["max-context"])
        except (OSError, KeyError, ValueError, TypeError):
            return None
    path = STRATA_CONFIGS.get(model_id)
    if path is None:
        return None
    try:
        args = json.loads(path.read_text())["args"]
        return int(args[args.index("--max-context") + 1])
    except (OSError, KeyError, ValueError, IndexError):
        return None

# pi defaults each model's maxTokens (max OUTPUT tokens, sent as the request's
# max_completion_tokens) to 16384, which truncates heavy thinking models such as
# Qwen3.8 mid-reasoning. Set it to this fraction of the model's context window
# instead, so the output cap is effectively removed. llama.cpp clamps n_predict
# to the space actually left after the prompt, so a value this high is safe.
MAX_TOKENS_RATIO = 0.95

# Per-family pi thinking/effort wiring. A spec matches a base model when EITHER
# its id (lowercased) contains one of `marker`, OR its cmd names one of
# `template`. Overrides reasoning / thinkingLevelMap / compat on the base entry
# (so hand-curated "off": null maps get replaced by real controls).
#
# Why both: the family is really determined by the chat template (that is what
# decides which kwargs are accepted), not by the model's name. Fine-tunes get
# named after themselves -- Swift-1.5-27B-Q6KS runs the Qwen3.8 sharp template
# but its id carries no q3.8/qwen3.8 marker, so marker-only matching silently
# stripped its thinking controls with no error at all.
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
    # Project Maya (maya-glm, maya-deepseek; maya/run-maya.sh). FIRST: maya-deepseek would
    # otherwise match the llama.cpp DeepSeek spec below, whose `thinking` kwarg Maya's
    # server ignores (thinking could not be turned off). Maya reads top-level or
    # chat_template_kwargs `reasoning_effort` (none/low/medium/high/xhigh/max, folded to
    # none/low/high/max: medium = high, xhigh = max) and `enable_thinking: false`
    # (serve/frontend.py, v1.1.42). Both configs default to "high" (2026-10-31).
    {
        "marker": "maya-",
        "label": "Maya",
        "defaultLevel": "high",
        "fields": {
            "reasoning": True,
            "thinkingLevelMap": {
                "minimal": None, "low": "low", "medium": None,
                "high": "high", "xhigh": "max", "max": "max",
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
    # Qwen3.8: full on/off + effort. Template accepts low/medium/high/xhigh
    # (qwen-3_8-improved-chat-template.jinja validates and raises otherwise).
    # `marker` may be a string OR a list of strings (see the matching loop in
    # main()). Both conventions are listed because config.yaml was renamed to short
    # ids on 2026-09-17 (Qwen3.8-* -> Q3.8-*); a marker that silently stops
    # matching strips pi's thinking controls with NO error at all.
    # `template` is the durable half of the match: every entry that runs this
    # jinja file takes reasoning_effort and enable_thinking, whatever it is
    # called (Hemmingway and Swift-1.5-27B-Q6KS were both being missed).
    {
        "marker": ["q3.8", "qwen3.8"],
        "template": "qwen_sharp_template_oneline.jinja",
        "label": "Qwen3.8",
        # pi's startup thinking level for every model this spec matches (see
        # PI_SETTINGS below). Optional; a spec without it leaves pi's global default.
        "defaultLevel": "medium",
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
        "label": "DeepSeek",
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
        "label": "Qwen3.5",
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
        "label": "Qwen3.6",
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
        "label": "Laguna",
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
    # Strata (ids Strata-IQ3S / subagent), the Qwen3.8-Flash-Next engine
    # behind a proxy. It is OpenAI-compatible and takes the standard top-level
    # `reasoning_effort` field, so thinkingFormat "openai" -- NOT the chat-template
    # mechanism used above, and no chat-template kwargs at all. It does not
    # understand the `developer` role, so pi must send `system`.
    # LEVELS (2026-10-03): Qwen3.8's chat template supports exactly low, medium and
    # xhigh (its default), and maps `high` to xhigh (raise_exception on anything else).
    # Checked live: Strata passes `xhigh` through. So pi shows off/low/medium/xhigh;
    # minimal and high are hidden -- they were only aliases for low and xhigh.
    {
        # 2026-11-01: Basalt (a Strata fork, Basalt-QFN = `subagent`) takes the same levels
        # (serve/frontend.py: none/low/medium/high->xhigh/xhigh) and the same OpenAI field.
        "marker": ["strata", "basalt"],
        "label": "Basalt",
        "defaultLevel": "medium",   # Strata and Basalt serve Qwen3.8-Flash-Next models
        "fields": {
            "reasoning": True,
            "thinkingLevelMap": {
                "off": "off", "minimal": None, "low": "low",
                "medium": "medium", "high": None,
                "xhigh": "xhigh", "max": None,
            },
            "compat": {
                "supportsDeveloperRole": False,
                "thinkingFormat": "openai",
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
            or re.search(r"--max-context\s+(\d+)", body)    # Strata / Project Maya (maya/run-maya.sh)
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


# Where the generated files land on every machine, as (local file, path under its ~).
SYNC_FILES = [
    (OUT, ".pi/agent/models.json"),
    (CODEX_CATALOG, ".codex/model-catalogs/marvin-models.json"),
    (CODEX_PROFILE, ".codex/marvin.config.toml"),
]
# ssdnodes runs ssh on port 42; the others on the default.
REMOTE_MACHINES = [
    ("ssdnodes", "ssd-nodes.akita-betelgeuse.ts.net", 42),
    ("guybrush", "guybrush.akita-betelgeuse.ts.net", None),
    ("colin", "colin.akita-betelgeuse.ts.net", None),
]


def push() -> None:
    """scp every SYNC_FILE to every REMOTE_MACHINE.

    scp will not create a directory, so each remote directory is mkdir -p'd first (the
    Codex one is new on most machines). BatchMode + ConnectTimeout keep a machine that is
    off the tailnet from hanging the whole sync: it fails with a warning instead.
    """
    print("sending to " + ", ".join(name for name, _, _ in REMOTE_MACHINES))
    for name, host, port in REMOTE_MACHINES:
        base = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        for local, rel in SYNC_FILES:
            parent = rel.rsplit("/", 1)[0]
            try:
                subprocess.run(["ssh"] + base + (["-p", str(port)] if port else [])
                               + [host, f"mkdir -p {parent}"],
                               check=True, capture_output=True, timeout=60)
                subprocess.run(["scp"] + base + (["-P", str(port)] if port else [])
                               + [str(local), f"{host}:~/{rel}"],
                               check=True, capture_output=True, timeout=120)
                print(f"  {name}: ~/{rel}")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                detail = ((e.stderr or b"").decode(errors="replace").strip().splitlines()
                          if isinstance(e, subprocess.CalledProcessError) else ["timed out"])
                print(f"  WARNING: {name}: ~/{rel} failed"
                      f"{' (' + detail[-1][:160] + ')' if detail else ''}"
                      f"; local files are written, copy manually")


def codex_profile_text() -> str:
    return f"""# `codex -p marvin` -- the llama-swap models on Marvin, written by ~/llama-swap/sync-models.py.
# Edit the generator, not this file: every run rewrites it. Personal Codex settings stay in
# config.toml -- this layer only sets what switching to Marvin actually means.
#
# model_catalog_json replaces the WHOLE catalog, so it belongs in this profile and never in the
# top of config.toml (there it would hide the OpenAI models). Generated file, no hand edits.
model = "subagent"
model_provider = "marvin"
model_catalog_json = "{CODEX_CATALOG_REL}"

# Self-contained on purpose: guybrush, ssd-nodes and colin each carry a slightly different
# marvin provider (or none), so the profile defines its own and every machine behaves the same.
[model_providers.marvin]
name = "Marvin"
base_url = "{PROVIDER["baseUrl"]}"
wire_api = "responses"
"""


def ensure_codex_profile() -> None:
    """Write ~/.codex/marvin.config.toml and get any legacy [profiles.marvin] out of config.toml.

    A missing catalog file is a HARD error for the profile, so the profile is written after the
    catalog (push() delivers both to the other machines).
    """
    try:
        CODEX_PROFILE.write_text(codex_profile_text())
    except OSError as e:
        sys.exit(f"error writing {CODEX_PROFILE}: {e}")
    print(f"wrote {CODEX_PROFILE}")
    try:
        text = CODEX_CONFIG.read_text()
    except OSError:
        return
    lines = text.splitlines(keepends=True)
    keep, dropped, inside = [], [], False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            now_inside = stripped.startswith("[profiles.marvin]")
            if now_inside and not inside:
                # the "# Profile for switching to Marvin ..." comment above it goes too
                while keep and keep[-1].lstrip().startswith("#"):
                    dropped.append(keep.pop().rstrip("\n"))
            inside = now_inside
        if inside or re.match(r'^profile\s*=\s*"marvin"', stripped):
            dropped.append(line.rstrip("\n"))
            continue
        keep.append(line)
    if not dropped:
        return
    try:
        backup = CODEX_CONFIG.with_name(f"config.toml.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        CODEX_CONFIG.write_text("".join(keep))
    except OSError as e:
        print(f"WARNING: could not remove the legacy marvin profile from {CODEX_CONFIG}: {e}")
        return
    print(f"moved the legacy marvin profile out of {CODEX_CONFIG} into {CODEX_PROFILE}")
    print(f"  removed {len(dropped)} line(s) ({dropped[0]!r} ...); previous file kept as {backup.name}")
    print("  codex -p marvin fails outright while [profiles.marvin] or profile = \"marvin\" is left in config.toml")


def codex_configured() -> bool:
    """True when a local Codex config actually points at the generated catalog."""
    for path in (CODEX_CONFIG, CODEX_PROFILE):
        try:
            if "model_catalog_json" in path.read_text():
                return True
        except OSError:
            continue
    return False


def match_spec(model_id: str, key: str, cmd: str) -> dict | None:
    """The THINKING_WIRING spec for a model: first match on an id/config-key marker
    or on the chat template named in the cmd wins (so order matters). Shared by the pi
    wiring pass and the Codex catalog so the two agents can never disagree about what a
    model accepts."""
    key = key.lower()
    for spec in THINKING_WIRING:
        markers = spec["marker"]
        if isinstance(markers, str):
            markers = [markers]
        templates = spec.get("template") or []
        if isinstance(templates, str):
            templates = [templates]
        if any(mk in model_id.lower() or mk in key for mk in markers) \
                or any(t in cmd for t in templates):
            return spec
    return None


def codex_reasoning(spec: dict | None) -> dict:
    """default_reasoning_level / supported_reasoning_levels for one Codex model entry.

    Both keys are REQUIRED together (a catalog without them fails to load: "missing field
    supported_reasoning_levels"), and default_reasoning_level may not be "". Codex sends the
    effort verbatim as `reasoning.effort`, so the accepted set is the spec's non-null
    thinkingLevelMap values -- what the template/engine takes -- intersected with the effort
    names Codex knows.

    A model with no spec, or an on/off-only family (Qwen3.5/3.6, Laguna: empty map), gets an
    empty list and the default "none": nothing to dial, so ask for no thinking. Templates that
    RAISE on an unknown effort (Qwen3.8) always come with a spec and a list, so they never see
    it. Note pi is not limited this way -- it drives enable_thinking plus chat_template_kwargs
    per level, which Codex's catalog cannot express.
    """
    level_map: dict = dict(((spec or {}).get("fields") or {}).get("thinkingLevelMap") or {})
    accepted = [e for e in CODEX_EFFORTS if e in set(level_map.values())]
    if not accepted:
        return {"default_reasoning_level": "none", "supported_reasoning_levels": []}
    default = (spec or {}).get("defaultLevel")
    if default not in accepted:
        default = accepted[0]
    return {
        "default_reasoning_level": default,
        "supported_reasoning_levels": [
            {"effort": e, "description": f"{e} reasoning effort"} for e in accepted
        ],
    }


def codex_catalog(models: list[dict], cmds: dict[str, str], keys: dict[str, str],
                  instructions: str) -> dict:
    """The same model list as pi's, in Codex's model_catalog_json shape.

    Field notes for anyone editing this:
    - context_window comes straight from the pi entry, i.e. out of the model's own -c /
      --max-context flag, so Codex's auto-compact threshold matches the real window instead of
      guessing (the reason this file exists).
    - include_apps_usage_instructions false: Apps/Connectors are ChatGPT cloud services, they
      are irrelevant here and cost thousands of tokens of prompt on every request. Skills stay
      on so ~/.codex/skills still work.
    - apply_patch_tool_type is deliberately unset: with it unset Codex makes edits through the
      shell, which every model here manages. Set it to "freeform" only if you want to test the
      patch tool on a model that handles it.
    """
    entries = []
    for priority, m in enumerate(models, start=1):
        spec = match_spec(m["id"], keys.get(m["id"], ""), cmds.get(m["id"], ""))
        ctx = int(m["contextWindow"])
        family = (spec or {}).get("label")
        entry = {
            "slug": m["id"],
            "display_name": m["id"],
            "description": f"{ctx // 1024}K context" + (f" \u00b7 {family}" if family else ""),
            "shell_type": "unified_exec",
            "visibility": "list",
            "supported_in_api": True,
            "priority": priority,
            "context_window": ctx,
            "support_verbosity": False,
            # reasoning.summary stays "auto" in the request whatever this says: it comes from
            # model_reasoning_summary in config.toml, not from the catalog (checked 2026-11-11).
            "truncation_policy": {"mode": "tokens", "limit": 10000},
            "experimental_supported_tools": [],
            "include_skills_usage_instructions": True,
            "include_plugin_usage_instructions": False,
            "include_apps_usage_instructions": False,
            "input_modalities": (["text", "image"]
                                 if any(k in m["id"].lower() for k in CODEX_IMAGE_INPUT_MARKERS)
                                 else ["text"]),
            **codex_reasoning(spec),
            "base_instructions": instructions,
        }
        entries.append(entry)
    return {"models": entries}


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
    # model id -> its cmd, so the wiring pass below can match on the chat
    # template as well as the id.
    cmds: dict[str, str] = {}
    keys: dict[str, str] = {}   # model id -> config key
    for name, entry in cfg["models"].items():
        if entry.get("unlisted") or name in EXCLUDES:
            continue
        aliases = model_aliases(entry)
        model_id = aliases[0] if aliases else name
        cmds[model_id] = str(entry.get("cmd") or "")
        keys[model_id] = name
        if len(aliases) > 1:
            # Only the first alias becomes the pi model id; further aliases stay
            # routable in llama-swap but would be duplicate entries here.
            print(f"note: {name}: extra aliases not exposed to pi:", aliases[1:])
        # Curated fields follow the model, not its name: look up the new id first
        # (that is what models.json used on the previous run), then the config
        # key, so renaming a model to an alias keeps its reasoning/thinking wiring.
        keep: dict[str, object] = dict(cur_models.get(model_id) or cur_models.get(name)
                                       or {"input": ["text"]})
        if model_id != name and name in cur_models and model_id not in cur_models:
            renames.append((name, model_id))
        keep["id"] = model_id
        # strata_context is keyed by the CONFIG KEY: the id can be an alias that
        # moves between models (subagent moved 27B -> Strata on 2026-09-30).
        parsed = (context_window(entry.get("cmd", "")) or strata_context(name)
                  or CONTEXT_OVERRIDES.get(model_id))
        if parsed is None:
            ctx_unparsed.append(name)
        ctx = parsed or keep.get("contextWindow") or DEFAULT_CONTEXT_WINDOW
        keep["contextWindow"] = ctx
        keep["maxTokens"] = max_tokens(ctx)
        new_models.append(keep)
        if model_id != name:
            # Also list the model under its config key, so pi shows the real model
            # (Strata-IQ3S) beside the alias (subagent). llama-swap routes
            # both names to the same process. The copy is taken after the curated
            # fields and context are settled, so the two entries never disagree.
            twin = {**keep, "id": name}
            new_models.append(twin)
            cmds[name] = cmds[model_id]
            keys[name] = name

    # Apply per-family thinking/effort wiring to the base entries (match_spec: marker
    # substring of the id OR config key, or the chat template named in the cmd; first
    # matching spec wins, so order matters). The config key counts because an alias id
    # says nothing about the family: `subagent` moved from a Qwen3.8 27B to Strata and
    # kept the 27B's chat-template wiring.
    levels: dict[str, str] = {}   # "marvin/<id>" -> pi startup thinking level
    for m in new_models:
        spec = match_spec(m["id"], keys.get(m["id"], ""), cmds.get(m["id"], ""))
        if spec:
            m.update(spec["fields"])
            if spec.get("defaultLevel"):
                levels[f"{PROVIDER_NAME}/{m['id']}"] = spec["defaultLevel"]

    out = {
        "providers": {
            PROVIDER_NAME: {**PROVIDER, "models": new_models}
        }
    }

    try:
        settings = json.loads(PI_SETTINGS.read_text()) if PI_SETTINGS.exists() else {}
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"error reading {PI_SETTINGS}: {e}")
    old_levels = dict(settings.get("modelThinkingLevels") or {})
    new_levels = {k: v for k, v in old_levels.items()
                  if not k.startswith(f"{PROVIDER_NAME}/")}
    new_levels.update(levels)

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
    level_changes = sorted(k for k in set(old_levels) | set(new_levels)
                           if old_levels.get(k) != new_levels.get(k))
    print("pi thinking-level changes:",
          [f"{k}: {old_levels.get(k)} -> {new_levels.get(k)}" for k in level_changes] or "none")

    # --- Codex catalog ---------------------------------------------------------
    try:
        instructions = CODEX_INSTRUCTIONS.read_text()
    except OSError as e:
        sys.exit(f"error reading {CODEX_INSTRUCTIONS} (Codex needs base_instructions): {e}")
    catalog = codex_catalog(new_models, cmds, keys, instructions)
    codex_ids = {e["slug"] for e in catalog["models"]}
    cur_codex_ids: set[str] = set()
    if CODEX_CATALOG.exists():
        try:
            cur_codex_ids = {e["slug"] for e in json.loads(CODEX_CATALOG.read_text())["models"]}
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as e:
            print(f"note: could not read the old {CODEX_CATALOG} ({e}); no Codex diff this run")

    print("codex removing:", sorted(cur_codex_ids - codex_ids) or "nothing")
    print("codex adding:  ", sorted(codex_ids - cur_codex_ids) or "nothing")

    if dry_run:
        print("(dry run — nothing written)")
        return

    text = json.dumps(out, indent=2) + "\n"
    OUT.write_text(text)
    shutil.copy2(OUT, DEST)
    print(f"wrote {OUT}")
    print(f"copied to {DEST}")
    if level_changes:
        settings["modelThinkingLevels"] = new_levels
        PI_SETTINGS.write_text(json.dumps(settings, indent=2) + "\n")
        print(f"updated modelThinkingLevels in {PI_SETTINGS}")

    try:
        CODEX_CATALOG.parent.mkdir(parents=True, exist_ok=True)
        CODEX_CATALOG.write_text(json.dumps(catalog, indent=2) + "\n")
    except OSError as e:
        sys.exit(f"error writing {CODEX_CATALOG}: {e}")
    print(f"wrote {CODEX_CATALOG} ({len(catalog['models'])} models)")
    ensure_codex_profile()
    if not codex_configured():
        print(f"note: no ~/.codex config sets model_catalog_json, so Codex is not reading "
              f"{CODEX_CATALOG} yet. Put it in the marvin PROFILE (~/.codex/marvin.config.toml) "
              f"-- in config.toml it would replace the whole catalog and hide the OpenAI models.")

    push()
    print("done")


if __name__ == "__main__":
    main()
