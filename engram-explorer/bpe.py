#!/usr/bin/env python3
"""
Minimal byte-level BPE encoder for Qwen3.8-Flash-Next (gpt2 byte-level / qwen35 pre),
built from the GGUF vocab + merges. No external tokenizer dependency.

Encoding pipeline (gpt2 byte-level):
  1. Byte->unicode char mapping (GPT-2 style printable remap).
  2. Pre-tokenize: split on whitespace, prefix each word with U+0120 ('Ġ').
  3. Encode each token with the merge rules (from tokenizer.ggml.merges).
  4. Look up resulting byte-sequences in the vocab to get token ids.

The merges from GGUF are stored as UTF-8 byte pairs separated by a literal
space (0x20). We convert each merge's two sides to the same "char-space" the
pre-tokenized words use (i.e. apply the same byte remap to both sides), so
merges match the encoded words directly.
"""

import numpy as np
from gguf import GGUFReader

# GPT-2 / byte-level BPE byte<->char mapping (canonical bytes_to_unicode)
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) \
        + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs)), dict(zip(cs, bs))

BYTE_TO_CHAR, CHAR_TO_BYTE = _bytes_to_unicode()
_SP = "\u0120"  # 'Ġ'


def bytes_to_chars(b: bytes) -> str:
    return "".join(BYTE_TO_CHAR[x] for x in b)


def chars_to_bytes(s: str) -> bytes:
    return bytes(CHAR_TO_BYTE[c] for c in s)


class ByteLevelBPE:
    def __init__(self, gguf_path):
        r = GGUFReader(gguf_path)
        # vocab tokens -> ids
        tparts = r.fields["tokenizer.ggml.tokens"].parts
        ntok = int(np.asarray(tparts[4])[0])
        token_bytes = []
        i = 6
        while len(token_bytes) < ntok and i < len(tparts) - 1:
            data = np.asarray(tparts[i]); i += 1
            if i >= len(tparts):
                break
            n = int(np.asarray(tparts[i])[0]); i += 1
            token_bytes.append(bytes(data[:n]))
        # map byte-string -> token id
        self.byte_to_id = {}
        for idx, tb in enumerate(token_bytes):
            if tb not in self.byte_to_id:
                self.byte_to_id[tb] = idx

        # merges: pair of char-space strings (both sides byte-remapped) -> rank
        mparts = r.fields["tokenizer.ggml.merges"].parts
        nm = int(np.asarray(mparts[4])[0])
        self.merges = {}
        rank = 0
        i = 6
        while rank < nm and i < len(mparts) - 1:
            data = np.asarray(mparts[i]); i += 1
            if i >= len(mparts):
                break
            n = int(np.asarray(mparts[i])[0]); i += 1
            pair_bytes = bytes(data[:n])
            # split on the first literal space byte
            sp = pair_bytes.find(0x20)
            if sp == -1:
                continue
            left = bytes_to_chars(pair_bytes[:sp])
            right = bytes_to_chars(pair_bytes[sp + 1:])
            key = (left, right)
            if key not in self.merges:
                self.merges[key] = rank
            rank += 1
        self._cache = {}

    def _encode_word(self, word: str):
        """word is a char-space string (already Ġ-prefixed if starts a token)."""
        # initial splits: each char as its own byte-seq
        parts = [[c] for c in word]
        while len(parts) > 1:
            best = None
            best_rank = None
            for j in range(len(parts) - 1):
                key = ("".join(parts[j]), "".join(parts[j + 1]))
                if key in self.merges:
                    rk = self.merges[key]
                    if best_rank is None or rk < best_rank:
                        best = j
                        best_rank = rk
            if best is None:
                break
            merged = "".join(parts[best]) + "".join(parts[best + 1])
            parts[best:best + 2] = [[merged]]
        return [chars_to_bytes("".join(p)) for p in parts]

    def encode(self, text: str) -> list:
        """text (unicode) -> list of token ids."""
        if not text:
            return []
        if text in self._cache:
            return list(self._cache[text])
        ids = []
        for w in _gpt2_pretokenize(text):
            # gpt2 byte-level: prepend Ġ to each word
            w_chars = _SP + bytes_to_chars(w.encode("utf-8"))
            for tb in self._encode_word(w_chars):
                tid = self.byte_to_id.get(tb)
                if tid is None:
                    for byte in tb:
                        tid = self.byte_to_id.get(bytes([byte]))
                        if tid is not None:
                            ids.append(tid)
                else:
                    ids.append(tid)
        self._cache[text] = ids
        return list(ids)


def _gpt2_pretokenize(text: str):
    """Split text on whitespace boundaries, keeping runs."""
    out = []
    cur = []
    for ch in text:
        if ch.isspace():
            if cur:
                out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


if __name__ == "__main__":
    bpe = ByteLevelBPE("/mnt/models/Qwen3.8-Flash-Next-UD-IQ4_XS/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf")
    import sys
    for phrase in sys.argv[1:] or ["The cat sat on the", "hello world", "attention is all you need"]:
        ids = bpe.encode(phrase)
        print(repr(phrase), "->", ids)
