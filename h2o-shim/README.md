# h2o-lightning-4b System 1 service

This folder serves the H2O-Lightning-4B decision model as H2O intended: a typed JSON
API that returns calibrated probabilities. It runs H2O's own shim over llama-server
(llama-swap entry `h2o-lightning-4b`) instead of vLLM.

```
client --POST /v1/systemone--> h2o-shim (pm2, :8741)
       --> llama-swap :8033/upstream/h2o-lightning-4b --> llama-server (ZOTAC, Q4_K_M GGUF)
```

| Item | Detail |
|---|---|
| `h2o_lightning_shim.py`, `serve_config.json` | H2O's files, **unmodified**, from `h2oai/h2o-lightning-4b` tag v1.2.2 (identical on main at 09d3d56, 2026-10-09). Licence Apache-2.0. sha256 c899c93d1b51f46d… / a423fd6b973334bd… |
| `llama_backend.py` | Replaces the shim's vLLM client with llama-server calls, then serves the shim as shipped |
| pm2 app | `h2o-shim`: `python3 llama_backend.py --host 0.0.0.0 --port 8741` (stdlib only, no venv) |
| Model | `/mnt/shared/models/h2o-lightning-4b.Q4_K_M.gguf` (mradermacher static quant, sha256 a2856d3e…). It loads on first use and then stays: it's in llama-swap's persistent "resident" group |

## API

These are H2O's own API, unchanged. See their model card for the full contract.

- `POST /v1/systemone` takes `{"state": <text or JSON>, "questions": {<id>: {"type": ..., "instructions": ..., "criteria": ...}}}`.
  - `choice`: `criteria` is `{name: description}`, up to 255 options.
  - `noul` (yes/no): `criteria` is `{"true": ..., "false": ...}`; an optional `threshold` adds `decision`.
  - `score`: `criteria` is an ordered list of levels.
- The response is `answers` with `probabilities` and `confidence`, plus `usage`.
- Malformed questions get a 422, an empty request a 400, and a backend fault a 502 or 503.
- `GET /health` reports the active settings (temperature 0.8, yes/no floor 0.801).

```sh
curl -s localhost:8741/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Ticket 4471: checkout returns HTTP 500 for every customer; no orders for 40 minutes.",
  "questions": {"all_users": {"type": "noul", "instructions": "The problem affects every customer.",
                              "criteria": {"true": "Affects everyone", "false": "Affects only some"}}}}'
```

## How it differs from H2O's setup

- **Label readout.** vLLM returns the log-probability of exactly the label tokens
  (`logprob_token_ids`). llama-server can't, so `llama_backend.py` reads the top 1000
  tokens of the raw distribution at the answer slot. A label outside that list gets
  the smallest value returned, which is an upper bound on its true value, and a line
  is written to the pm2 log. Only options with a negligible share are affected.
- **Q4_K_M weights** instead of bf16 with an fp32 output head.
- **No images.** The GGUF has no vision projector. Image support needs an mmproj
  GGUF, made from the HF weights with `convert_hf_to_gguf.py --mmproj`, plus a
  chat-path backend.
- **Context.** llama-server runs with `-c 16384`; H2O's vLLM setup has 40,960. Longer
  records get a 422 here, though the shim itself truncates only above 32,000 tokens.
- **Cosmetic.** `/health` still says `backend: vllm ...`, because the text comes from
  H2O's file.

## Verified (2026-10-09)

H2O's model-card example (ticket 4471, three questions) gives:

| Question | H2O's result (vLLM, bf16) | Here |
|---|---|---|
| priority (choice) | p1, probabilities 0.944 / 0.040 / 0.017, confidence 0.916 | p1, probabilities 0.949 / 0.034 / 0.017, confidence 0.924 |
| all_users (yes/no) | 0.929 | 0.937 |
| urgency (score) | 1.789 | 1.797 |
| input tokens | 231 | 231 |

The identical token count shows the prompt is rendered exactly as H2O's own setup
renders it.
