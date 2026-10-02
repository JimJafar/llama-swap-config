#!/usr/bin/env python3
"""
Token-continuation explorer backend for Qwen3.8-Flash-Next.

Serves two things over plain HTTP (stdlib http.server):
  1. GET  /search?q=<text>[&limit=N]  -> token dropdown: tokens whose decoded
     text partially matches <text>. Returns [{id, text, token, byte_len}, ...].
  2. GET  /continue?t=<tok1>,<tok2>,..[&k=N]  -> top-K next-token continuations
     for the given token sequence, fetched live from the llama.cpp server's
     n_probs. Returns [{id, text, token, logprob}, ...] plus the composed
     prefix text.

The llama.cpp upstream is reached via llama-swap (model
Qwen3.8-Flash-Next-UD-IQ4_XS-explore, --reasoning off) so continuations are
substantive tokens, not chain-of-thought noise.

Env:
  LLAMA_SWAP_URL   base URL of llama-swap (default http://127.0.0.1:8033)
  MODEL            explore model id (default Qwen3.8-Flash-Next-UD-IQ4_XS-explore)
  GGUF_PATH        shard 1 of the model (default the one next to this script's host)
  EXPLORER_PORT    listen port (default 8350)
  EXPLORER_HOST    bind address (default 0.0.0.0; 127.0.0.1 behind Tailscale Serve)
"""

import http.server
import json
import os
import re
import sys
import urllib.parse
import urllib.request

from gguf import GGUFReader
import numpy as np

from bpe import ByteLevelBPE

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
LLAMA_SWAP_URL = os.environ.get("LLAMA_SWAP_URL", "http://127.0.0.1:8033")
# Direct llama-server for the explore instance (llama-swap only proxies /v1/*, not
# /tokenize or /completion). Port comes from the config's -p ${PORT}:8080 mapping.
DIRECT_URL = os.environ.get("DIRECT_URL", "http://127.0.0.1:5804")
MODEL = os.environ.get("MODEL", "Qwen3.8-Flash-Next-UD-IQ4_XS-explore")
GGUF_PATH = os.environ.get(
    "GGUF_PATH",
    "/mnt/models/Qwen3.8-Flash-Next-UD-IQ4_XS/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf",
)
PORT = int(os.environ.get("EXPLORER_PORT", "8350"))
HOST = os.environ.get("EXPLORER_HOST", "0.0.0.0")

# --------------------------------------------------------------------------
# tokenizer: load vocab from GGUF, decode byte-level BPE to readable text
# --------------------------------------------------------------------------
class Tokenizer:
    def __init__(self, path):
        self.bpe = ByteLevelBPE(path)
        r = GGUFReader(path)
        # Each string in a GGUF array is stored as (length, bytes); field.data holds the
        # indices of the byte parts, so read those rather than pairing parts by hand
        # (the old loop paired each token with the NEXT token's length and truncated it).
        field = r.fields["tokenizer.ggml.tokens"]
        self.tokens = [bytes(np.asarray(field.parts[idx])) for idx in field.data]
        # decode each token string (gpt2 byte-level) to readable text
        self.texts = [self._decode(t) for t in self.tokens]
        # full vocab list of dicts for search
        self.vocab = [
            {"id": idx, "text": self.texts[idx],
             "token": self.tokens[idx].decode("utf-8", errors="replace"),
             "byte_len": len(self.tokens[idx])}
            for idx in range(len(self.tokens))
        ]
        # byte-length filter (exclude pure punctuation / specials in dropdown later)
        # raw token string (as llama.cpp reports it) -> id, for resolving n_probs
        self.raw_to_id = {}
        for idx, tb in enumerate(self.tokens):
            key = tb.decode("utf-8", errors="replace")
            if key not in self.raw_to_id:
                self.raw_to_id[key] = idx

    def resolve_id(self, raw: str):
        """Map a raw token string (as llama.cpp reports it, e.g. 'ĠHello' or 'Hello')
        to its id. llama-server strips the space marker, so try direct then with
        the 'Ġ' (U+0120) prefix, which is how the vocab stores word-initial tokens."""
        if isinstance(raw, int):
            return raw if 0 <= raw < len(self.tokens) else None
        if not raw:
            return None
        if raw in self.raw_to_id:
            return self.raw_to_id[raw]
        # word-initial token: llama reports 'Hello', vocab has 'ĠHello'
        cand = "\u0120" + raw
        if cand in self.raw_to_id:
            return self.raw_to_id[cand]
        return None

    def decode_ids(self, ids):
        """Decode a token sequence as one byte string, so characters split across
        tokens (emoji, CJK) come out whole instead of as replacement characters."""
        raw = bytearray()
        for tok in ids:
            if not 0 <= tok < len(self.tokens):
                continue
            text = self.tokens[tok].decode("utf-8", errors="replace")
            if all(ch in _BYTE_DECODER for ch in text):
                raw += bytes(_BYTE_DECODER[ch] for ch in text)
            else:
                raw += text.encode("utf-8")
        return raw.decode("utf-8", errors="replace")

    @staticmethod
    def _decode(b):
        # Qwen uses GPT-2 byte-level BPE: every byte is stored as a printable stand-in
        # character (space -> 'Ġ', newline -> 'Ċ', 0xE6 -> 'æ', ...). Map each character
        # back to its byte, then decode UTF-8; a token holding part of a multi-byte
        # character shows U+FFFD, as llama.cpp's own pieces do.
        text = b.decode("utf-8", errors="replace")
        if all(ch in _BYTE_DECODER for ch in text):
            return bytes(_BYTE_DECODER[ch] for ch in text).decode("utf-8", errors="replace")
        return text  # special tokens such as <|im_start|> are stored as plain text


def _gpt2_byte_decoder():
    """Inverse of GPT-2's bytes_to_unicode table (the stand-in character for each byte)."""
    printable = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\u00a1"), ord("\u00ac") + 1))
                 + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    codes, extra = list(printable), 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codes.append(256 + extra)
            extra += 1
    return {chr(code): byte for byte, code in zip(printable, codes)}


_BYTE_DECODER = _gpt2_byte_decoder()


# --------------------------------------------------------------------------
# continuation query to llama-swap
# --------------------------------------------------------------------------
# set in main(); used by get_continuations
_tokenizer = None


def warmup():
    """Ensure the explore model is loaded. llama-swap lazy-loads a model only on an
    actual inference request, so send a tiny completion through it (returns once the
    model is up and the direct llama-server port responds)."""
    try:
        req = urllib.request.Request(
            LLAMA_SWAP_URL + "/v1/completions",
            data=json.dumps({"model": MODEL, "prompt": "Hi", "max_tokens": 1,
                             "stream": False}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=240)
    except Exception:
        pass


def llama_tokenize(text):
    """Use the direct llama-server /tokenize for authoritative text->token ids."""
    warmup()
    req = urllib.request.Request(
        DIRECT_URL + "/tokenize",
        data=json.dumps({"content": text, "add_special": False}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp).get("tokens", [])


def get_continuations(token_ids, k=8):
    """Fetch top-K next-token logprobs for a token sequence.
    Sends the raw token ids so the model does NOT re-tokenize (avoids drift).
    Uses the /completion endpoint with prompt=<ids>, raw=true, n_predict=1."""
    t = _tokenizer
    readable = t.decode_ids(token_ids)
    body = {
        "prompt": token_ids,
        "raw": True,
        "n_predict": 1,
        "stream": False,
        "n_probs": k,
        "temperature": 0.0,
    }
    warmup()
    req = urllib.request.Request(
        DIRECT_URL + "/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    out = []
    # /completion with n_probs returns completion_probabilities[].top_logprobs[]
    # where each item carries {id, token, bytes, logprob}.
    for entry in (data.get("completion_probabilities") or []):
        for item in (entry.get("top_logprobs") or []):
            tid = item.get("id")
            tok = item.get("token")
            lp = item.get("logprob")
            if lp is None:
                continue
            # decoded readable text: use the model's reported token (already decoded),
            # but prefer our vocab text for a clean byte-space rendering when available
            text = t.texts[tid] if isinstance(tid, int) and 0 <= tid < len(t.texts) else str(tok)
            out.append({
                "token": tok,
                "text": text,
                "id": tid if isinstance(tid, int) else -1,
                "logprob": round(lp, 4) if lp is not None else None,
            })
    return {"prefix": readable, "continuations": out}


# --------------------------------------------------------------------------
# static frontend
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _static_handler(path):
    # only allow known files from this dir
    if path in ("/", "/index.html"):
        f = os.path.join(BASE_DIR, "index.html")
    elif path == "/app.css":
        f = os.path.join(BASE_DIR, "app.css")
    elif path == "/app.js":
        f = os.path.join(BASE_DIR, "app.js")
    else:
        return None
    if os.path.isfile(f):
        ctype = "text/html" if f.endswith(".html") else ("text/css" if f.endswith(".css") else "application/javascript")
        with open(f, "rb") as fh:
            return ctype, fh.read()
    return None


# --------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------
def build_handler(tokenizer):
    def handle(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        # static first
        st = _static_handler(parsed.path)
        if st is not None:
            ctype, body = st
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        try:
            if parsed.path == "/search":
                q = (qs.get("q", [""])[0]).strip()
                limit = int(qs.get("limit", ["50"])[0])
                self._json(200, search_tokens(tokenizer, q, limit))
            elif parsed.path == "/continue":
                toks = qs.get("t", [""])[0]
                k = int(qs.get("k", ["8"])[0])
                ids = [int(x) for x in toks.split(",") if x]
                result = get_continuations(ids, k)
                self._json(200, result)
            elif parsed.path == "/continue_text":
                text = qs.get("text", [""])[0]
                k = int(qs.get("k", ["8"])[0])
                ids = llama_tokenize(text)
                result = get_continuations(ids, k)
                result["tokens"] = [
                    {"id": i, "text": tokenizer.texts[i] if 0 <= i < len(tokenizer.texts) else str(i)}
                    for i in ids
                ]
                self._json(200, result)
            elif parsed.path == "/encode":
                text = qs.get("text", [""])[0]
                ids = llama_tokenize(text)
                self._json(200, {"text": text, "tokens": [
                    {"id": i, "text": tokenizer.texts[i] if 0 <= i < len(tokenizer.texts) else str(i)} for i in ids
                ]})
            elif parsed.path == "/health":
                self._json(200, {"ok": True, "vocab": len(tokenizer.tokens)})
            else:
                self._json(404, {"error": "not found"})
        except Exception as e:
            self._json(500, {"error": str(e)})
    return type("Handler", (http.server.BaseHTTPRequestHandler,), {
        "do_GET": handle,
        "do_POST": handle,
        "_json": _json_response,
        "log_message": lambda *a, **k: None,
    })


def _json_response(self, code, payload):
    body = json.dumps(payload).encode()
    self.send_response(code)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Access-Control-Allow-Origin", "*")
    self.end_headers()
    self.wfile.write(body)


def search_tokens(tokenizer, q, limit=50):
    """Return vocab tokens whose decoded text contains q (substring)."""
    if not q:
        return {"query": "", "results": []}
    ql = q.lower()
    results = []
    # prefer prefix matches first, then substring; skip empty/whitespace-only text
    for entry in tokenizer.vocab:
        t = entry["text"]
        if not t or not t.strip():
            continue
        tl = t.lower()
        if tl.startswith(ql):
            entry["match"] = "prefix"
            results.append(entry)
            if len(results) >= limit:
                return {"query": q, "results": results}
    for entry in tokenizer.vocab:
        t = entry["text"]
        if not t or not t.strip():
            continue
        tl = t.lower()
        if ql in tl:
            entry["match"] = "substring"
            results.append(entry)
            if len(results) >= limit:
                break
    return {"query": q, "results": results}


def main():
    global _tokenizer
    tk = Tokenizer(GGUF_PATH)
    _tokenizer = tk
    httpd = http.server.ThreadingHTTPServer((HOST, PORT), build_handler(tk))
    print(f"engram-explorer backend on {HOST}:{PORT} (vocab={len(tk.tokens)})", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
