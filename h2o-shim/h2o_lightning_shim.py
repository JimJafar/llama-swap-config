#!/usr/bin/env python3
"""H2O-Lightning-4B shim: typed decisions over a stock vLLM server, one output token per decision. Stdlib only.

WHAT IT DOES
    POST /v1/systemone  (the `typesafe` decision wire format; /api/alpha/decisions is an alias)
      -> renders each question in the prompt the model was trained on (serve_config.json: the chat template with
         thinking off, the system prompt, the "plain" layout, compact JSON for a structured state, one label per
         option, the "Answer:" prefill)
      -> sends ONE vLLM completion per question with max_tokens=1 and `logprob_token_ids` set to that question's
         label tokens, so every option is read at the answer slot and none can fall out of a top-k
      -> applies one temperature for every question type, then the yes/no commit floor
      -> returns {"model", "answers", "probability_source", "effort_used", "usage"}

    Several questions about one state are sent concurrently; vLLM's prefix cache shares the state between them, and
    usage.input_tokens counts that shared head once plus each question's own tail.
    It generates no text and calls nothing but the local vLLM server.

    python3 h2o_lightning_shim.py --config serve_config.json --vllm http://127.0.0.1:8000 --port 8741

IMAGES (only when serve_config.json has an "image" block, and only for a request that carries an image)
    Accepted forms: a `data:image/...;base64,` URI or bare base64 image bytes anywhere inside `state` (a string, an
    object value, a list item); a chat-style image part inside `state` ({"type": "image_url", "image_url": {"url"}},
    {"type": "image", "image": ...}, {"type": "input_image", "image_url": ...}, {"type": "base64", "media_type",
    "data"}); the top-level fields `images`, `image`, `image_url`, `image_urls`, `image_data`, `image_base64` (data
    URIs, bare base64, http(s) URLs, or the part objects above); and multipart/form-data with the JSON in a
    `request` field and the images as file parts. A URL is an image only in those explicit places, never as a plain
    string inside `state`. Each image lifted out of `state` leaves `[image N]` in the record (the top-level images are
    numbered first). The question is then read through vLLM's chat completions with the images ahead of the same
    user text, the assistant turn pre-filled with the same "Answer:" (continue_final_message) and the same
    `logprob_token_ids` readout, so the prompt is the text prompt plus the image tokens. A request without an image
    takes the text path above, unchanged.

STATUS CODES (an evaluation runner stops after three consecutive failures unless a failure is a 422): a request this
system cannot answer -- too many options, a malformed question, a prompt over vLLM's context -- is a 422, scored as
one wrong answer; an empty request is a 400; an unreachable or failing vLLM is a 502, and a vLLM that does not serve
the configured model is a 503, so a broken backend stops a run; vLLM's own 401/403/429 pass through.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import email.parser
import email.policy
import http.client
import socket
import json
import math
import os
import re
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SHIM_VERSION = "h2o-lightning-shim"
PRIMITIVES = ("choice", "noul", "score")
EFFORT_TIERS = ("fast", "balanced", "thorough")
# vLLM 0.30 accepts at most 128 `logprob_token_ids` per request; a question with more options is read in chunks
LOGPROB_CHUNK = int(os.environ.get("SHIM_LOGPROB_CHUNK", "128"))
TIMEOUT = float(os.environ.get("SHIM_TIMEOUT", "180"))
# vLLM's wording when a prompt does not fit --max-model-len (it has changed between releases)
_CONTEXT_ERROR = re.compile(r"maximum context length|maximum model length|max_model_len|too long and exceeds|"
                            r"longer than the maximum", re.I)
# Qwen3.5's chat template for a system and a user message with the generation prompt and thinking off; the template
# trims both contents (Jinja's `trim` is Python's str.strip)
CHATML = {"system": "<|im_start|>system\n{content}<|im_end|>\n",
          "user": "<|im_start|>user\n{content}<|im_end|>\n",
          "generation": "<|im_start|>assistant\n<think>\n\n</think>\n\n"}
# what the template writes for one image part, before vLLM expands the pad into the image's tokens
VISION_SLOT = "<|vision_start|><|image_pad|><|vision_end|>"
MAX_IMAGE_BYTES = int(os.environ.get("SHIM_MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))


class Unprocessable(ValueError):
    """A request in the contract's shape that this system cannot answer: a 422, one wrong item, the run goes on."""


class BadRequest(ValueError):
    """Not the contract's shape at all (no questions, an unknown effort tier): a 400."""


class UpstreamError(RuntimeError):
    """vLLM failed or could not be reached: a 502 (or 503, or vLLM's own 401/403/429), so a run stops."""

    def __init__(self, message, status=502):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------------------------------------- the prompt
def render_state(state, compact):
    """The record: a string verbatim, anything else as JSON (compact when the config says so)."""
    if isinstance(state, str):
        return state
    if compact:
        return json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(state, indent=1)


def desc_text(v):
    """An option description or score level: a string verbatim, anything else as JSON."""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def render_instr(value):
    """`instructions` may be a string, an object or an array."""
    return value if isinstance(value, str) else json.dumps(value, indent=1)


def options_of(question, max_options=255):
    """(option names, descriptions) for the three question types. A yes/no question is always [true, false]."""
    typ = question.get("type", "choice")
    criteria = question.get("criteria")
    if typ not in PRIMITIVES:
        raise Unprocessable(f"question type {typ!r} is not one of {list(PRIMITIVES)}")
    if typ == "noul":
        criteria = criteria or {}
        if not isinstance(criteria, dict):
            raise Unprocessable("noul criteria must be an object with true/false descriptions")
        true_desc = criteria.get("true", criteria.get("yes"))
        false_desc = criteria.get("false", criteria.get("no"))
        return ["true", "false"], [
            desc_text(true_desc) if true_desc not in (None, "") else "the statement holds",
            desc_text(false_desc) if false_desc not in (None, "") else "it does not",
        ]
    if typ == "score":
        if not isinstance(criteria, list) or len(criteria) < 1:
            raise Unprocessable("score criteria must be an ordered array of levels")
        return [str(i) for i in range(len(criteria))], [desc_text(value) for value in criteria]
    if not isinstance(criteria, dict) or len(criteria) < 1:
        raise Unprocessable("choice criteria must map option -> description")
    if len(criteria) > max_options:
        raise Unprocessable(f"choice supports at most {max_options} options")
    names = list(criteria)
    return names, [desc_text(criteria[name]) if criteria[name] is not None else name for name in names]


def option_lines(labels, names, descs):
    """One line per option: `A) name: description`."""
    if len(labels) < len(names):
        raise Unprocessable(f"{len(names)} options but only {len(labels)} labels")
    return [f"{l}) {n}: {d}" for l, n, d in zip(labels, names, descs)]


def user_turn(state_text, instructions, names, descs, labels, layout="plain"):
    """The user message."""
    body = "\n".join(option_lines(labels, names, descs))
    if layout == "plain":
        return f"record: {state_text}\nquestion: {instructions}\noptions:\n{body}"
    if layout == "markdown":
        return f"### RECORD\n{state_text}\n\n### QUESTION\n{instructions}\n\n### OPTIONS\n{body}\n"
    raise ValueError(f"layout {layout!r} is not supported")


def render_chat(system, user, prefill, chat=True):
    """The full prompt: the chat template for [system, user] with thinking off, then the answer prefill. Without the
    template, the same parts joined by blank lines."""
    if not chat:
        return "\n\n".join(([] if system is None else [system]) + [user]) + f"\n{prefill}"
    out = "" if system is None else CHATML["system"].replace("{content}", system.strip())
    return out + CHATML["user"].replace("{content}", user.strip()) + CHATML["generation"] + prefill


def chat_messages(system, user, prefill, image_urls):
    """The same prompt as render_chat, as chat messages for vLLM: the images ahead of the user text, and the assistant
    turn pre-filled with the prefill (sent with continue_final_message). Qwen3.5's template renders a final assistant
    turn after the last user turn as `<think>\\n\\n</think>\\n\\n` + its content, which is render_chat's generation
    prompt + prefill, so with no image the two token streams are the same (checked against vLLM at first use)."""
    msgs = [] if system is None else [{"role": "system", "content": system.strip()}]
    parts = [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
    msgs.append({"role": "user", "content": parts + [{"type": "text", "text": user.strip()}]})
    msgs.append({"role": "assistant", "content": prefill})
    return msgs


# ------------------------------------------------------------------------------------------------- images
IMAGE_KEYS = ("images", "image", "image_url", "image_urls", "image_data", "image_base64")
_DATA_URI = re.compile(r"data:(image/[A-Za-z0-9.+-]+);base64,(.*)", re.S | re.I)
_B64_CHARS = re.compile(r"[A-Za-z0-9+/=\s]+")
_B64_URLSAFE = re.compile(r"[A-Za-z0-9_\-=\s]+")
_PART_TYPES = ("image_url", "image", "input_image")
# how base64 of a PNG, JPEG, GIF and WebP begins: the cheap test before anything inside `state` is decoded
_B64_HEADS = ("iVBORw0KGgo", "/9j/", "R0lGOD", "UklGR")


def sniff_image(raw):
    """The image type from its first bytes, or None."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith(b"BM"):
        return "image/bmp"
    if raw[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    return None


def _b64_bytes(s):
    """Decoded bytes of a (standard or URL-safe) base64 string, or None."""
    t = re.sub(r"\s+", "", s)
    if not t:
        return None
    try:
        if _B64_CHARS.fullmatch(t):
            return base64.b64decode(t + "=" * (-len(t) % 4), validate=True)
        if _B64_URLSAFE.fullmatch(t):
            return base64.urlsafe_b64decode(t + "=" * (-len(t) % 4))
    except (binascii.Error, ValueError):
        return None
    return None


def _data_uri(mime, raw):
    if len(raw) > MAX_IMAGE_BYTES:
        raise Unprocessable(f"an image of {len(raw)} bytes is over the {MAX_IMAGE_BYTES}-byte limit")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def image_from_string(s, explicit):
    """A string as an image URL for vLLM, or None when it is not an image.

    A `data:image/...;base64,` URI is always an image (a malformed one is a 422). Bare base64 counts only when it
    decodes to known image bytes; inside `state` it must also be one unbroken token of at least 64 characters that
    decodes to a PNG, JPEG, GIF or WebP, so ordinary text never qualifies. An http(s) URL counts only when `explicit`
    (a top-level image field or an image part): a record may hold ordinary links."""
    s = s.strip()
    m = _DATA_URI.match(s)
    if m:
        raw = _b64_bytes(m.group(2))
        if not raw:
            raise Unprocessable("an image data URI does not carry valid base64")
        return _data_uri(sniff_image(raw) or m.group(1).lower(), raw)
    if explicit:
        if re.match(r"https?://", s, re.I):
            return s
        raw = _b64_bytes(s) if len(s) >= 16 else None
        mime = sniff_image(raw) if raw else None
        if mime:
            return _data_uri(mime, raw)
        raise Unprocessable("an image field holds neither a data URI, base64 image bytes nor an http(s) URL")
    if len(s) < 64 or not _B64_CHARS.fullmatch(s) or any(c.isspace() for c in s) or not s.startswith(_B64_HEADS):
        return None
    raw = _b64_bytes(s)
    mime = sniff_image(raw) if raw else None
    return _data_uri(mime, raw) if mime in ("image/png", "image/jpeg", "image/gif", "image/webp") else None


def image_from_part(obj, in_state=False):
    """A chat/API-style image object as an image URL, or None when `obj` is not one. Inside `state` (`in_state`) an
    object shaped like a part whose value is not an image (say {"type": "image", "image": "a sunset"}) is record
    data and stays; at the top level it is a 422."""
    if not isinstance(obj, dict):
        return None
    typ = obj.get("type")
    if typ == "base64" and isinstance(obj.get("data"), str):              # {"type": "base64", "media_type", "data"}
        v = f"data:{obj.get('media_type') or 'image/png'};base64,{obj['data']}"
    elif typ in _PART_TYPES:
        v = obj.get(typ) if typ != "input_image" else obj.get("image_url", obj.get("image"))
        if isinstance(v, dict):
            v = v.get("url", v.get("data"))
        if not isinstance(v, str):
            if in_state:
                return None
            raise Unprocessable(f"an image part of type {typ!r} has no url or data")
    else:
        return None
    if in_state and not (_DATA_URI.match(v.strip()) or re.match(r"https?://", v.strip(), re.I)):
        return image_from_string(v, False)
    return image_from_string(v, True)


def explicit_image(v):
    """One entry of a top-level image field: a string or an image object ({"url"}, {"image_url": {"url"}}, a part)."""
    if isinstance(v, str):
        return image_from_string(v, True)
    if isinstance(v, dict):
        got = image_from_part(v)
        if got is not None:
            return got
        for k in ("url", "image_url", "data", "image"):
            inner = v.get(k)
            if isinstance(inner, dict):
                inner = inner.get("url", inner.get("data"))
            if isinstance(inner, str):
                return image_from_string(inner, True)
    raise Unprocessable("an image entry must be a string or an object with a url or base64 data")


def lift_images(state, found):
    """`state` with every image inside it replaced by `[image N]` (N counts on from len(found)); the images are
    appended to `found`. Anything that is not an image is returned as is (the same object when nothing changed)."""
    if isinstance(state, str):
        u = image_from_string(state, False)
        if u is None:
            return state
        found.append(u)
        return f"[image {len(found)}]"
    if isinstance(state, dict):
        u = image_from_part(state, in_state=True)
        if u is not None:
            found.append(u)
            return f"[image {len(found)}]"
        out, changed = {}, False
        for k, v in state.items():
            out[k] = lift_images(v, found)
            changed = changed or out[k] is not v
        return out if changed else state
    if isinstance(state, list):
        out = [lift_images(v, found) for v in state]
        return out if any(a is not b for a, b in zip(out, state)) else state
    return state


def extract_images(body):
    """(state with `[image N]` placeholders, [image URLs]) for a request; (the same state, []) when it has none."""
    found = []
    for k in IMAGE_KEYS:
        v = body.get(k)
        if v is None:
            continue
        for item in (v if isinstance(v, list) else [v]):
            found.append(explicit_image(item))
    return lift_images(body["state"], found), found


def parse_multipart(content_type, raw):
    """A multipart/form-data body as a request object: the JSON from the `request` (or `body`, `json`, `data`) field,
    every file part as an image in the top-level `images` list, after any images the JSON already has."""
    msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + raw)
    if not msg.is_multipart():
        raise Unprocessable("a multipart request without parts")
    body, files = None, []
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        data = part.get_payload(decode=True) or b""
        if part.get_filename() is not None or (part.get_content_maintype() == "image"):
            mime = sniff_image(data) or (part.get_content_type() if part.get_content_maintype() == "image" else None)
            if not mime:
                raise Unprocessable(f"file part {name!r} is not a recognised image")
            files.append(_data_uri(mime, data))
        elif name in ("request", "body", "json", "data"):
            try:
                body = json.loads(data.decode("utf-8"))
            except ValueError as e:
                raise Unprocessable(f"multipart field {name!r} is not JSON: {e}") from e
    if not isinstance(body, dict):
        raise Unprocessable("a multipart request needs its JSON in a `request` field")
    if files:
        have = body.get("images")
        body["images"] = (have if isinstance(have, list) else ([] if have is None else [have])) + files
    return body


# ------------------------------------------------------------------------------------------------- the answer
def confidence(p):
    """Normalized maximum: 1 for a point mass, 0 for a uniform distribution."""
    n = len(p)
    if n < 2:
        return 1.0
    return float((max(p) - 1.0 / n) / (1.0 - 1.0 / n))


def probabilities(logprobs, temperature):
    """softmax(log-probabilities / T) over the options."""
    if not logprobs or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                               for v in logprobs):
        raise UpstreamError("vLLM returned an invalid log-probability vector")
    v = [x / max(float(temperature), 1e-3) for x in logprobs]
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    if not math.isfinite(s) or s <= 0:
        raise UpstreamError("vLLM returned log-probabilities that cannot be normalized")
    return [x / s for x in e]


def commit_noul(p_yes, floor):
    """A yes/no probability with max(P, 1-P) under the floor moves to the floor on its own side (a tie goes to yes).
    The answer never changes; 0 turns the floor off."""
    if not floor:
        return p_yes
    if max(p_yes, 1.0 - p_yes) >= floor:
        return p_yes
    return floor if p_yes >= 0.5 else 1.0 - floor


def noul_threshold(q):
    """A yes/no question's optional decision threshold in [0, 1], else None."""
    if q.get("type", "choice") != "noul":
        return None
    t = q.get("threshold", q.get("noul_threshold"))
    return float(t) if isinstance(t, (int, float)) and 0.0 <= float(t) <= 1.0 else None


def format_answer(typ, names, descs, logprobs, temperature, noul_floor, threshold=None):
    """One question's answer in the wire shape."""
    p = probabilities(logprobs, temperature)
    if typ == "noul":
        py = commit_noul(float(p[0]), noul_floor)
        answer = {"type": "noul", "noul": float(py)}
        if threshold is not None:
            answer["threshold"] = float(threshold)
            answer["decision"] = bool(py >= float(threshold))
        return answer
    if typ == "score":
        return {
            "type": "score",
            "score": float(sum(i * x for i, x in enumerate(p))),
            "legend": {str(i): d for i, d in enumerate(descs)},
            "probabilities": {str(i): float(x) for i, x in enumerate(p)},
            "confidence": confidence(p),
        }
    k = max(range(len(p)), key=p.__getitem__)          # the first maximum
    return {
        "type": "choice",
        "choice": names[k],
        "probabilities": {name: float(x) for name, x in zip(names, p)},
        "confidence": confidence(p),
    }


class Contract:
    """The serving settings, from serve_config.json."""

    def __init__(self, cfg, env=os.environ):
        p = cfg["prompt"]
        self.cfg = cfg
        self.model = cfg["model"]
        self.chat = bool(p.get("chat_template", True))
        if p.get("enable_thinking", False):
            raise ValueError("prompt.enable_thinking must be false: the answer is read after a closed think block")
        self.system = p.get("system") or None
        self.layout = p.get("layout", "plain")
        self.state_compact = bool(p.get("state_compact", True))
        self.prefill = p.get("prefill", "Answer:")
        self.max_state_tokens = int(p.get("max_state_tokens", 32000))
        self.default_instructions = p.get("default_instructions", "Answer the question below.")
        if not isinstance(cfg.get("labels"), list):
            raise ValueError("serve_config.json: `labels` must be the list of option labels, in order")
        self.labels = [str(l) for l in cfg["labels"]][: int(cfg.get("max_options", len(cfg["labels"])))]
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("serve_config.json: the labels are not distinct")
        self.label_ids = None             # each label's token id at the answer slot, read from vLLM's tokenizer
        # SHIM_TEMPERATURE / SHIM_NOUL_FLOOR override the config; SHIM_NOUL_FLOOR=0 turns the floor off
        t_env, f_env = env.get("SHIM_TEMPERATURE", "").strip(), env.get("SHIM_NOUL_FLOOR", "").strip()
        self.temperature = float(t_env) if t_env else float(cfg["temperature"])
        self.temperature_source = "SHIM_TEMPERATURE" if t_env else "serve_config.json"
        self.noul_floor = float(f_env) if f_env else float(cfg["noul_floor"])
        self.noul_floor_source = "SHIM_NOUL_FLOOR" if f_env else "serve_config.json"
        if not 0.05 <= self.temperature <= 20.0:
            raise ValueError(f"temperature {self.temperature} is outside [0.05, 20]")
        if self.noul_floor and not 0.5 < self.noul_floor < 1.0:
            raise ValueError(f"noul_floor {self.noul_floor}: must be in (0.5, 1), or 0 to turn it off")
        # IMAGES: on only with an "image" block. SHIM_MAX_IMAGES / SHIM_MAX_PIXELS / SHIM_IMAGE_TEMPERATURE override it;
        # max_pixels goes to vLLM's image processor with every image request (0: the server's own setting)
        im = cfg.get("image")
        self.images = isinstance(im, dict)
        im = im if self.images else {}
        mi, mp, it = (env.get(k, "").strip() for k in ("SHIM_MAX_IMAGES", "SHIM_MAX_PIXELS", "SHIM_IMAGE_TEMPERATURE"))
        self.max_images = int(mi) if mi else int(im.get("max_images", 4))
        self.max_pixels = int(mp) if mp else int(im.get("max_pixels", 0) or 0)
        self.image_temperature = float(it) if it else float(im.get("temperature", self.temperature))
        if self.images and not self.chat:
            raise ValueError("images need prompt.chat_template: the image path renders through the chat template")
        if self.images and not (1 <= self.max_images <= 16 and 0.05 <= self.image_temperature <= 20.0):
            raise ValueError("image.max_images must be in [1, 16] and image.temperature in [0.05, 20]")

    def question_parts(self, state_text, q, qid="q"):
        """(user text, option names, descriptions, labels) for one question."""
        names, descs = options_of(q, len(self.labels))
        if len(names) > len(self.labels):
            raise Unprocessable(f"question {qid}: {len(names)} options exceeds {len(self.labels)}")
        labels = self.labels[:len(names)]
        instr = render_instr(q.get("instructions", "")).lstrip() or self.default_instructions
        return user_turn(state_text, instr, names, descs, labels, self.layout), names, descs, labels

    def question_prompt(self, state_text, q, qid="q"):
        """(prompt text, option names, descriptions, labels) for one question."""
        user, names, descs, labels = self.question_parts(state_text, q, qid)
        return render_chat(self.system, user, self.prefill, self.chat), names, descs, labels

    def image_prompt(self, state_text, q, image_urls, qid="q"):
        """(chat messages, the same prompt as text with one VISION_SLOT per image, names, descriptions, labels)."""
        user, names, descs, labels = self.question_parts(state_text, q, qid)
        text = render_chat(self.system, VISION_SLOT * len(image_urls) + user.strip(), self.prefill, self.chat)
        return chat_messages(self.system, user, self.prefill, image_urls), text, names, descs, labels


# ------------------------------------------------------------------------------------------------- vLLM client
QUICKACK = getattr(socket, "TCP_QUICKACK", None)   # Linux; reset by the kernel, so set per request


class VLLM:
    """A keep-alive HTTP/1.1 connection per thread to one vLLM server."""

    def __init__(self, base, model):
        u = urllib.parse.urlsplit(base if "://" in base else "http://" + base)
        self.base = f"{u.scheme}://{u.netloc}"
        self.https = u.scheme == "https"
        self.host = u.netloc
        self.model = model
        self.local = threading.local()

    def _conn(self, fresh=False):
        c = getattr(self.local, "conn", None)
        if c is None or fresh:
            if c is not None:
                c.close()
            cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
            c = self.local.conn = cls(self.host, timeout=TIMEOUT)
        return c

    def call(self, method, path, body=None):
        """(status, parsed JSON or text). A connection the server closed is reopened once."""
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"} if data is not None else {}
        for attempt in (0, 1):
            try:
                c = self._conn(fresh=attempt == 1)
                c.request(method, path, body=data, headers=headers)
                if QUICKACK and c.sock is not None:   # ACK vLLM's reply at once: a kept-alive connection would
                    c.sock.setsockopt(socket.IPPROTO_TCP, QUICKACK, 1)   # otherwise delay it (~40 ms with Nagle)
                r = c.getresponse()
                raw = r.read()
                break
            except (http.client.HTTPException, OSError) as e:
                if attempt == 1:
                    raise UpstreamError(f"vLLM at {self.base} unreachable: {type(e).__name__}: {e}") from e
        try:
            return r.status, json.loads(raw)
        except ValueError:
            return r.status, raw.decode("utf-8", "replace")

    def post(self, path, body, item_errors=(400,)):
        status, out = self.call("POST", path, body)
        if status == 200 and isinstance(out, dict):
            return out
        detail = json.dumps(out)[:500] if isinstance(out, dict) else str(out)[:500]
        if status in (400, 413) and _CONTEXT_ERROR.search(detail):
            raise Unprocessable(f"over the server's context limit: {detail}")
        if status in item_errors:
            raise Unprocessable(f"vLLM refused the request: {detail}")
        raise UpstreamError(f"vLLM {path} HTTP {status}: {detail}", status if status in (401, 403, 429) else 502)

    def tokenize(self, text):
        return self.post("/tokenize", {"model": self.model, "prompt": text, "add_special_tokens": False})["tokens"]

    def detokenize(self, ids):
        return self.post("/detokenize", {"model": self.model, "tokens": ids})["prompt"]

    def label_logprobs(self, prompt, ids):
        """{token id: log-probability} at the answer slot, one completion per LOGPROB_CHUNK labels."""
        out, usage = {}, None
        for k in range(0, len(ids), LOGPROB_CHUNK):
            chunk = ids[k:k + LOGPROB_CHUNK]
            r = self.post("/v1/completions", {
                "model": self.model, "prompt": prompt, "max_tokens": 1, "temperature": 0.0,
                "logprobs": len(chunk), "logprob_token_ids": chunk, "return_tokens_as_token_ids": True,
                "add_special_tokens": False})
            try:
                top = r["choices"][0]["logprobs"]["top_logprobs"][0]
                got = {int(key.split(":", 1)[1]): float(v) for key, v in top.items()}
            except (KeyError, IndexError, TypeError, ValueError) as e:
                raise UpstreamError(f"vLLM returned no label log-probabilities at the answer slot ({e})") from e
            missing = [i for i in chunk if i not in got]
            if missing:
                raise UpstreamError(f"vLLM omitted label tokens {missing[:5]} from logprob_token_ids")
            out.update({i: got[i] for i in chunk})
            usage = usage or r.get("usage") or {}
        return out, usage

    def tokenize_chat(self, messages):
        """vLLM's token ids for chat messages rendered as the image path renders them (no image parts here)."""
        return self.post("/tokenize", {"model": self.model, "messages": messages, "add_generation_prompt": False,
                                       "continue_final_message": True, "add_special_tokens": False,
                                       "chat_template_kwargs": {"enable_thinking": False}})["tokens"]

    def label_logprobs_chat(self, messages, ids, max_pixels=0):
        """label_logprobs for a chat with images: the final assistant turn (the prefill) is continued, so the answer
        slot is the same position as in the text prompt. One chat completion per LOGPROB_CHUNK labels."""
        out, usage = {}, None
        for k in range(0, len(ids), LOGPROB_CHUNK):
            chunk = ids[k:k + LOGPROB_CHUNK]
            body = {"model": self.model, "messages": messages, "max_completion_tokens": 1, "temperature": 0.0,
                    "logprobs": True, "logprob_token_ids": chunk, "return_tokens_as_token_ids": True,
                    "add_generation_prompt": False, "continue_final_message": True, "add_special_tokens": False,
                    "chat_template_kwargs": {"enable_thinking": False}}
            if max_pixels:
                body["mm_processor_kwargs"] = {"max_pixels": int(max_pixels)}
            # an image vLLM cannot load (a URL it cannot fetch, bytes it cannot decode) is a 422 from vLLM: one bad item
            r = self.post("/v1/chat/completions", body, item_errors=(400, 422))
            try:
                top = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                got = {int(e["token"].split(":", 1)[1]): float(e["logprob"]) for e in top}
            except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
                raise UpstreamError(f"vLLM returned no label log-probabilities at the answer slot ({e})") from e
            missing = [i for i in chunk if i not in got]
            if missing:
                raise UpstreamError(f"vLLM omitted label tokens {missing[:5]} from logprob_token_ids")
            out.update({i: got[i] for i in chunk})
            usage = usage or r.get("usage") or {}
        return out, usage


def common_prefix_len(seqs):
    """The length of the longest common prefix of several token id lists; 0 for fewer than two."""
    if len(seqs) < 2:
        return 0
    n = min(len(x) for x in seqs)
    for i in range(n):
        if any(x[i] != seqs[0][i] for x in seqs[1:]):
            return i
    return n


# ------------------------------------------------------------------------------------------------------ server
class Shim:
    def __init__(self, contract, vllm, min_context=4096):
        self.c = contract
        self.v = vllm
        self.min_context = min_context
        self.ready = None                 # the backend check, done once at first use (vLLM may start later)
        self.ready_lock = threading.Lock()
        self.max_model_len = None
        self.chat_ok = None               # the image path's template check, done once at the first image request
        self.pool = ThreadPoolExecutor(max_workers=int(os.environ.get("SHIM_QUESTION_THREADS", "64")))

    def check_backend(self):
        """vLLM serves the configured model with a usable context, and every label is ONE token at the answer slot:
        after a probe prompt with all the labels, appending " <label>" adds exactly one token, whose id is read."""
        if self.ready:
            return
        with self.ready_lock:
            if self.ready:
                return
            status, out = self.v.call("GET", "/v1/models")
            models = (out.get("data") or []) if isinstance(out, dict) else []
            entry = next((m for m in models if isinstance(m, dict) and m.get("id") == self.c.model), None)
            if status != 200 or entry is None:
                raise UpstreamError(f"vLLM at {self.v.base} does not serve {self.c.model!r} "
                                    f"(start it with --served-model-name {self.c.model})", 503)
            self.max_model_len = entry.get("max_model_len")
            if isinstance(self.max_model_len, int) and self.max_model_len < self.min_context:
                raise UpstreamError(f"vLLM's max_model_len {self.max_model_len} is below {self.min_context}", 503)
            n = len(self.c.labels)
            probe = {"type": "choice", "instructions": "Which?", "criteria": {f"o{i}": f"d{i}" for i in range(n)}}
            text = self.c.question_prompt(render_state({"x": 1}, self.c.state_compact), probe)[0]
            base = self.v.tokenize(text)
            ids, bad = {}, []
            for label in self.c.labels:
                full = self.v.tokenize(text + " " + label)
                if len(full) != len(base) + 1 or full[:len(base)] != base:
                    bad.append(label)
                else:
                    ids[label] = full[-1]
            if bad or len(set(ids.values())) != len(ids):
                raise UpstreamError(f"these labels are not one distinct token at the answer slot under vLLM's "
                                    f"tokenizer: {bad[:8]} (is this serve_config.json the one for this model?)", 503)
            self.c.label_ids = ids
            self.ready = True
            sys.stderr.write(f"shim: vLLM serves {self.c.model} (max_model_len {self.max_model_len}); "
                             f"{n} labels verified\n")

    def truncate_state(self, text):
        """A record over max_state_tokens keeps its head and its tail around a "..." line (vLLM's tokenizer)."""
        mx = self.c.max_state_tokens
        if not mx or len(text.encode("utf-8", "ignore")) <= mx:      # a byte-level BPE never has more tokens than bytes
            return text
        ids = self.v.tokenize(text)
        if len(ids) <= mx:
            return text
        half = mx // 2
        return self.v.detokenize(ids[:half]) + "\n...\n" + self.v.detokenize(ids[-half:])

    def one(self, prompt, labels):
        ids = [self.c.label_ids[l] for l in labels]
        lp, usage = self.v.label_logprobs(prompt, ids)
        return [lp[i] for i in ids], usage

    def decide(self, body):
        """The /v1/systemone handler. Raises BadRequest / Unprocessable / UpstreamError."""
        t0 = time.time()
        if not isinstance(body, dict) or "state" not in body or not isinstance(body.get("questions"), dict):
            raise Unprocessable("expected a JSON object with `state` and a `questions` object")
        questions = body["questions"]
        if not questions:
            raise BadRequest("no questions")
        if not all(isinstance(q, dict) for q in questions.values()):
            raise Unprocessable("every question must be an object")
        effort = body.get("effort")
        if effort is not None and not isinstance(effort, (str, dict)):
            raise Unprocessable("effort must be a tier name or an object")
        asked = effort if isinstance(effort, str) or effort is None else effort.get("tier", "fast")
        if asked is not None and asked not in EFFORT_TIERS:
            raise BadRequest(f"effort tier {asked!r} is not one of {list(EFFORT_TIERS)}")
        self.check_backend()
        if self.c.images:                  # an image deployment: a request that carries an image takes the image path
            state, images = extract_images(body)
            if images:
                return self.decide_images(questions, state, images, asked, t0)
        state_txt = self.truncate_state(render_state(body["state"], self.c.state_compact))
        work = []
        for qid, q in questions.items():
            text, names, descs, labels = self.c.question_prompt(state_txt, q, qid)
            work.append((qid, q.get("type", "choice"), names, descs, text, labels, noul_threshold(q)))
        shared = 0
        if len(work) == 1:
            reads = [self.one(work[0][4], work[0][5])]
        else:                              # concurrently: vLLM batches them and its prefix cache shares the state
            futs = [self.pool.submit(self.one, w[4], w[5]) for w in work]
            toks = [self.pool.submit(self.v.tokenize, w[4]) for w in work]
            reads = [f.result() for f in futs]
            # COUNT THE SHARED HEAD ONCE: the system prompt and the record are computed once by vLLM's prefix
            # cache, so N whole prompts would bill the record N times
            shared = common_prefix_len([f.result() for f in toks])
        return self.respond(work, reads, shared, asked, t0, self.c.temperature)

    def check_chat(self):
        """Once, before the first image request: vLLM's chat template renders our messages (with no image) to exactly
        the tokens of the text prompt, so the image path differs from the trained prompt only by the image tokens."""
        if self.chat_ok:
            return
        with self.ready_lock:
            if self.chat_ok:
                return
            probe = {"type": "choice", "instructions": "Which?", "criteria": {"a": "first", "b": "second"}}
            user, _n, _d, _l = self.c.question_parts(render_state({"x": 1}, self.c.state_compact), probe)
            want = self.v.tokenize(render_chat(self.c.system, user, self.c.prefill, self.c.chat))
            got = self.v.tokenize_chat(chat_messages(self.c.system, user, self.c.prefill, []))
            if got != want:
                k = common_prefix_len([got, want]) if got and want else 0
                raise UpstreamError(f"vLLM's chat template does not render the trained prompt (tokens differ from "
                                    f"position {k} of {len(want)}); is this the model's chat_template.jinja?", 503)
            self.chat_ok = True
            sys.stderr.write("shim: image path verified (chat template renders the trained prompt)\n")

    def one_chat(self, messages, labels):
        ids = [self.c.label_ids[l] for l in labels]
        lp, usage = self.v.label_logprobs_chat(messages, ids, self.c.max_pixels)
        return [lp[i] for i in ids], usage

    def decide_images(self, questions, state, images, asked, t0):
        """The image path: the same record and question text, the images ahead of it, read through chat completions."""
        if len(images) > self.c.max_images:
            raise Unprocessable(f"{len(images)} images; this deployment reads at most {self.c.max_images} per request")
        self.check_chat()
        state_txt = self.truncate_state(render_state(state, self.c.state_compact))
        work, msgs = [], []
        for qid, q in questions.items():
            m, text, names, descs, labels = self.c.image_prompt(state_txt, q, images, qid)
            msgs.append(m)
            work.append((qid, q.get("type", "choice"), names, descs, text, labels, noul_threshold(q)))
        shared = 0
        if len(work) == 1:
            reads = [self.one_chat(msgs[0], work[0][5])]
        else:
            futs = [self.pool.submit(self.one_chat, m, w[5]) for m, w in zip(msgs, work)]
            toks = [self.pool.submit(self.v.tokenize, w[4]) for w in work]
            reads = [f.result() for f in futs]
            toks = [f.result() for f in toks]
            # the shared head holds the images: their tokens are what vLLM counted beyond the one-slot text prompt
            extra = max(0, int(reads[0][1].get("prompt_tokens") or 0) - len(toks[0]))
            shared = common_prefix_len(toks) + extra
        return self.respond(work, reads, shared, asked, t0, self.c.image_temperature, len(images))

    def respond(self, work, reads, shared, asked, t0, temperature, n_images=0):
        """The response: each answer at `temperature` (then the yes/no floor), and usage with the shared head once."""
        answers, ntok = {}, 0
        for (qid, typ, names, descs, _t, _l, thr), (lp, usage) in zip(work, reads):
            answers[qid] = format_answer(typ, names, descs, lp, temperature, self.c.noul_floor, thr)
            ntok += int(usage.get("prompt_tokens") or 0)
        ntok -= shared * (len(work) - 1)   # = the shared head + each question's tail
        used = {"tier": "fast", "passes_per_decision": 1}
        if asked and asked != "fast":
            used["requested"] = asked
            used["note"] = (f"tier {asked!r} is accepted for API compatibility but not implemented; this response "
                            f"was produced at tier 'fast' with one forward pass per decision")
        out = {"model": self.c.model, "answers": answers, "probability_source": "native", "effort_used": used,
               "usage": {"input_tokens": ntok, "output_tokens": 0, "latency_ms": round((time.time() - t0) * 1000, 1)}}
        if n_images:
            out["usage"]["images"] = n_images
        return out

    def health(self):
        c = self.c
        return {"ok": bool(self.ready), "model": c.model, "shim": SHIM_VERSION, "backend": f"vllm {self.v.base}",
                "layout": c.layout, "chat_template": c.chat, "state_compact": c.state_compact, "prefill": c.prefill,
                "max_state_tokens": c.max_state_tokens,
                "noul_floor": {"value": c.noul_floor, "source": c.noul_floor_source},
                "temperature_by_type": {t: c.temperature for t in PRIMITIVES},
                "temperature_source": c.temperature_source, "max_options": len(c.labels),
                "vllm_max_model_len": self.max_model_len,
                **({"images": {"max_images": c.max_images, "max_pixels": c.max_pixels or None,
                               "temperature": c.image_temperature, "verified": bool(self.chat_ok)}}
                   if c.images else {})}


def make_handler(shim):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # TCP_NODELAY on each client connection: the headers and the body go out as two writes, and on a kept-alive
        # connection Nagle would hold the body until the client's (delayed, ~40 ms) ACK of the headers
        disable_nagle_algorithm = True

        def log_message(self, fmt, *args):
            if os.environ.get("SHIM_ACCESS_LOG", "") == "1":
                sys.stderr.write("shim %s\n" % (fmt % args))

        def _send(self, code, obj):
            payload = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path.startswith("/health"):
                try:
                    shim.check_backend()
                except (UpstreamError, Unprocessable) as e:
                    return self._send(503, {**shim.health(), "error": str(e)})
                return self._send(200, shim.health())
            if self.path.startswith("/v1/models"):
                return self._send(200, {"object": "list", "data": [{"id": shim.c.model, "readout": "native label "
                                                                    "log-probabilities over vLLM"}]})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.startswith(("/v1/systemone", "/api/alpha/decisions")):
                return self._send(404, {"error": "not found"})
            ctype = self.headers.get("Content-Type") or ""
            try:
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n)
                if shim.c.images and ctype.lower().startswith("multipart/form-data"):
                    body = parse_multipart(ctype, raw)
                else:
                    body = json.loads(raw or b"null")
            except Unprocessable as e:
                return self._send(422, {"detail": str(e)})
            except (ValueError, OSError) as e:
                return self._send(422, {"detail": f"request body is not JSON: {e}"})
            try:
                return self._send(200, shim.decide(body))
            except BadRequest as e:
                return self._send(400, {"detail": str(e)})
            except Unprocessable as e:
                return self._send(422, {"detail": str(e)})
            except UpstreamError as e:
                return self._send(e.status, {"detail": str(e)})
            except Exception as e:   # one item's failure is a 422, never a 500
                import traceback
                traceback.print_exc()
                return self._send(422, {"detail": f"{type(e).__name__}: {str(e)[:300]}"})

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 512      # the standard library's listen backlog of 5 resets concurrent clients


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=os.environ.get("SHIM_CONFIG", os.path.join(here, "serve_config.json")))
    ap.add_argument("--vllm", default=os.environ.get("SHIM_VLLM", "http://127.0.0.1:8000"), help="vLLM base URL")
    ap.add_argument("--model", default=os.environ.get("SHIM_MODEL"),
                    help="vLLM's --served-model-name (default: serve_config.json's `model`)")
    ap.add_argument("--host", default=os.environ.get("SHIM_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("SHIM_PORT", "8741")))
    ap.add_argument("--min-context", type=int, default=int(os.environ.get("SHIM_MIN_CONTEXT", "4096")))
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config))
    if a.model:
        cfg["model"] = a.model
    contract = Contract(cfg)
    shim = Shim(contract, VLLM(a.vllm, contract.model), a.min_context)
    srv = Server((a.host, a.port), make_handler(shim))
    print(f"shim on {a.host}:{a.port} -> {a.vllm} (model {contract.model}, T {contract.temperature}, "
          f"yes/no floor {contract.noul_floor}, {len(contract.labels)} labels)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
