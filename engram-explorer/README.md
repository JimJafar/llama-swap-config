# EnGram Explorer

A token-continuation browser for **Qwen3.8-Flash-Next** (qwen4exp). It lets you:

1. **Type** in a dropdown to find tokens by partial match (searches the model's 248k-token vocab).
2. **Select** a token → the live model returns its top-K next-token continuations (with log-probs).
3. **Click** a continuation to append it and get the *next* set of continuations — chain indefinitely.

## What it shows (and what it doesn't)

Continuations are the model's real top-K next-token distribution, fetched live from a
dedicated **`--reasoning off`** llama.cpp server via its `/completion` + `n_probs` path
(temperature 0). Reasoning is off so the continuations are substantive tokens rather than
chain-of-thought noise (`We`/`Hmm`/`User`/...).

**Important honest caveat:** this explores the model's *output distribution*, not the PLE/engram
table itself. The 51 GB `per_layer_token_embd` table is a learned *hash codebook* (320M rows ×
160-dim), read at one transformer layer and gated internally — it is not a searchable
token-to-token index, and its per-token contribution is not observable from the server API.
What this tool gives you is the model's most-likely next token given a prefix, which is the
practical "what follows" answer.

## Architecture

```
Browser ──► backend.py (port 8350) ──► llama.cpp explore server (port 5804)
              │   /search        vocab substring search (from GGUF)
              │   /encode        llama /tokenize
              │   /continue?t=   llama /completion (raw token ids, n_probs)
              └── static index.html/app.css/app.js
```

- **backend.py** — stdlib `http.server` app (no deps beyond `gguf` + `numpy`), loads the GGUF
  vocab once, serves the dropdown search + continuation APIs + the frontend.
- **index.html / app.css / app.js** — the UI.
- **bpe.py** — *legacy* hand-rolled BPE encoder. **Not used** by the backend; the backend uses
  llama.cpp's native `/tokenize` (authoritative, context-correct). Kept only for reference.

## Run it

```bash
# backend (runs under pm2 as "engram-explorer")
cd ~/llama-swap/engram-explorer
pm2 start backend.py --name engram-explorer --interpreter python3
# then open http://<host>:8350/
```

Env vars (defaults shown):
- `LLAMA_SWAP_URL=http://127.0.0.1:8033` — llama-swap (used only to wake/load the model)
- `DIRECT_URL=http://127.0.0.1:5804` — the explore llama-server (for /tokenize + /completion)
- `MODEL=Qwen3.8-Flash-Next-UD-IQ4_XS-explore`
- `GGUF_PATH=/mnt/models/.../00001-of-00003.gguf`
- `EXPLORER_PORT=8350`

The model is lazy-loaded (cold start ~45 s on first request), then stays warm at ~180 ms per
continuation while resident. It holds all 3 GPUs (~44 GB), so it evicts/coexists with the other
llama-swap models per the usual "one big model at a time" rule.

## API

| Endpoint | Params | Returns |
|---|---|---|
| `GET /search?q=&limit=` | `q` partial text | `[{id, text, match}]` vocab matches |
| `GET /encode?text=` | text | `{tokens:[{id,text}]}` via llama `/tokenize` |
| `GET /continue?t=&k=` | comma-sep token ids | `{prefix, continuations:[{id,text,logprob}]}` |
| `GET /continue_text?text=&k=` | text | encode + continue in one call |
| `GET /health` | — | `{ok, vocab}` |
