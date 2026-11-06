"""Marvin GPU guard (2026-11-06): ComfyUI shares the 3090 + ZOTAC 5060 Ti with llama-swap.

ComfyUI runs on the 3090 (cuda:0, the diffusion model) and the ZOTAC (cuda:1, the text encoder),
the cards the default GLM (maya-glm) holds. Before ComfyUI queues a prompt, this unloads every
llama-swap model running on those two cards and waits for their VRAM, so the job has the cards.
Basalt (5070 Ti + MSI) is left alone. The other direction is maya/free-comfyui.sh, which the
models on these cards run before they start: it waits for ComfyUI's queue and frees its VRAM.

Which models use the cards is read from llama-swap's config.yaml at each prompt (any model whose
cmd names one of the two GPU UUIDs), so the list follows the config. Source lives in
~/llama-swap/comfyui/marvin-gpu-guard and is symlinked into ComfyUI's custom_nodes.
"""
import json
import logging
import re
import time
import urllib.request

from server import PromptServer

LLAMA_SWAP = "http://127.0.0.1:8033"
CONFIG = "/home/jim/llama-swap/config.yaml"
UUIDS = ("GPU-93ab52e5", "GPU-af5bd53f")   # the 3090, the ZOTAC
FREE_GIB = 6.0                             # wait until each card has at least this much free
TIMEOUT_S = 120

log = logging.getLogger("marvin-gpu-guard")


def models_on_cards():
    names, name = set(), None
    for line in open(CONFIG, encoding="utf-8"):
        m = re.match(r"^  ([A-Za-z0-9._-]+):\s*$", line)
        if m:
            name = m.group(1)
        elif name and not line.lstrip().startswith("#") and any(u in line for u in UUIDS):
            names.add(name)
    return names


def running():
    with urllib.request.urlopen(f"{LLAMA_SWAP}/running", timeout=5) as r:
        return {m["model"] for m in json.load(r).get("running", [])}


def cards_free_gib():
    import torch
    return [torch.cuda.mem_get_info(i)[0] / 2**30 for i in range(min(2, torch.cuda.device_count()))]


def guard(json_data):
    try:
        busy = running() & models_on_cards()
        if busy:
            log.info("unloading %s from the 3090 + ZOTAC for this prompt", ", ".join(sorted(busy)))
            for m in busy:
                req = urllib.request.Request(f"{LLAMA_SWAP}/api/models/unload/{m}", method="POST")
                urllib.request.urlopen(req, timeout=TIMEOUT_S).read()
        t = time.time()
        while time.time() - t < TIMEOUT_S:
            free = cards_free_gib()
            if not (running() & busy) and all(f >= FREE_GIB for f in free):
                break
            time.sleep(1)
        else:
            log.warning("cards still busy after %d s (free GiB: %s); queuing anyway", TIMEOUT_S, free)
    except Exception as e:   # never block a prompt on the guard itself
        log.warning("guard skipped: %s", e)
    return json_data


PromptServer.instance.add_on_prompt_handler(guard)
NODE_CLASS_MAPPINGS = {}
