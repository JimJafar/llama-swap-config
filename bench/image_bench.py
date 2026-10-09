#!/usr/bin/env python3
"""ComfyUI image speed test: FLUX.2 Klein 9B and Qwen Image 2.1, two images each.

Usage: bench/image_bench.py [--base http://127.0.0.1:8190] [--out bench/results/image.jsonl]

For each model, ComfyUI first unloads everything (POST /free), then renders
  1. 1024x1024  -- COLD: includes loading the diffusion model, text encoder and VAE
  2. 1920x1088  -- WARM: models already loaded (same prompt, so the text encoding is cached)
Time per image is ComfyUI's own execution_start -> execution_success. Graphs are the ones
used day to day (Klein: ~/comfyui/workflows/flux2-klein-9b.json; Qwen: the official
"Qwen Image 2.1: Text to Image" template, prompt enhancer off), with fixed prompts and seeds.
Images are saved in ComfyUI/output as hwbench-*.
"""
import argparse
import json
import pathlib
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
SIZES = [(1024, 1024), (1920, 1088)]
SEED = 20261009

KLEIN_PROMPT = ("A lighthouse on a rocky headland at dusk, painted in bold flat gouache shapes, "
                "limited palette of teal, rust and cream, visible brush texture")
QWEN_PROMPT = ("A quiet harbour town at dawn seen from a hillside: terracotta roofs, a stone "
               "breakwater with a small red lighthouse, fishing boats at their moorings, mist "
               "lying on the water, warm low sunlight, detailed realistic photograph")


def klein(w, h, tag):
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "Flux2-Klein-9B-True-v2-Q8_0.gguf"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_8b_fp4mixed.safetensors",
                                                     "type": "flux2", "device": "default"}},
        "14": {"class_type": "SelectCLIPDevice", "inputs": {"clip": ["2", 0], "device": "gpu:1"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2_ae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["14", 0], "text": KLEIN_PROMPT}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["4", 0],
                                                    "negative": ["5", 0], "cfg": 1.0}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "8": {"class_type": "Flux2Scheduler", "inputs": {"steps": 8, "width": w, "height": h}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "10": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["9", 0], "guider": ["6", 0],
                                                                 "sampler": ["7", 0], "sigmas": ["8", 0],
                                                                 "latent_image": ["10", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": tag}},
    }


def qwen(w, h, tag):
    return {
        "451": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors",
                                                       "weight_dtype": "default"}},
        "453": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors",
                                                       "type": "qwen_image", "device": "default"}},
        "454": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "480": {"class_type": "QwenImage21Cache", "inputs": {"model": ["451", 0], "device": "auto",
                                                             "dtype": "default"}},
        "452": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["453", 0], "vae": ["454", 0],
                                                                  "prompt": QWEN_PROMPT, "negative_prompt": "",
                                                                  "resolution": 1024}},
        "456": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "458": {"class_type": "KSampler", "inputs": {"model": ["480", 0], "positive": ["452", 0],
                                                     "negative": ["452", 1], "latent_image": ["456", 0],
                                                     "seed": SEED, "steps": 25, "cfg": 1.0,
                                                     "sampler_name": "euler", "scheduler": "simple",
                                                     "denoise": 1.0}},
        "457": {"class_type": "VAEDecode", "inputs": {"samples": ["458", 0], "vae": ["454", 0]}},
        "459": {"class_type": "SaveImage", "inputs": {"images": ["457", 0], "filename_prefix": tag}},
    }


MODELS = {"flux2-klein-9b (8 steps)": klein, "qwen-image-2.1 (25 steps)": qwen}


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read() or b"{}")


def run(base, graph):
    pid = post(base, "/prompt", {"prompt": graph, "client_id": "hwbench"})["prompt_id"]
    t0 = time.time()
    while True:
        time.sleep(1)
        with urllib.request.urlopen(f"{base}/history/{pid}", timeout=30) as r:
            h = json.loads(r.read())
        if pid in h and h[pid]["status"].get("completed") is not None:
            st = h[pid]["status"]
            ts = {m[0]: m[1].get("timestamp") for m in st.get("messages", [])}
            if st.get("status_str") != "success":
                raise RuntimeError(f"{st.get('status_str')}: {st.get('messages')}")
            return round((ts["execution_success"] - ts["execution_start"]) / 1000, 1)
        if time.time() - t0 > 1800:
            raise TimeoutError(pid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8190")
    ap.add_argument("--out", default=str(HERE / "results" / "image.jsonl"))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for name, build in MODELS.items():
        post(a.base, "/free", {"unload_models": True, "free_memory": True})
        time.sleep(5)
        rec = {"model": name, "date": time.strftime("%Y-%m-%d %H:%M"), "seed": SEED}
        for i, (w, h) in enumerate(SIZES):
            label = f"{w}x{h} " + ("cold" if i == 0 else "warm")
            rec[label] = run(a.base, build(w, h, f"hwbench-{name.split()[0]}-{w}x{h}"))
        print(json.dumps(rec), flush=True)
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
