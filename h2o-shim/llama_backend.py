#!/usr/bin/env python3
"""H2O's h2o_lightning_shim.py, unchanged, over llama-server (through llama-swap) instead of vLLM.

The shim talks to its backend only through its VLLM class. This file swaps that class for one that speaks
llama-server's API, then serves the shim as shipped: the same prompt, labels, temperature, yes/no floor and
JSON contract (POST /v1/systemone).

    python3 llama_backend.py --upstream http://127.0.0.1:8033/upstream/h2o-lightning-4b --port 8741

THE ONE DIFFERENCE, the label readout. vLLM returns the log-probability of exactly the label tokens
(`logprob_token_ids`); llama-server has no such option. Here /completion returns the top --n-probs tokens of the
raw (pre-sampling, temperature-free) distribution at the answer slot, and the labels are read from that. A label
outside the top --n-probs gets the smallest log-probability returned, an upper bound on its true value. At the
default 1000 that happens only to options with a negligible share, and is counted in the log (see --n-probs).

Images are off: the GGUF has no vision projector, so the config's "image" block is dropped.
"""
import argparse
import json
import os
import sys
import urllib.parse

import h2o_lightning_shim as shim

HERE = os.path.dirname(os.path.abspath(__file__))


class LlamaServer(shim.VLLM):
    """The shim's backend calls, answered by llama-server. `base` may carry a path prefix (llama-swap's
    /upstream/<model>), which every request is sent under."""

    n_probs = 1000

    def __init__(self, base, model):
        super().__init__(base, model)
        self.prefix = urllib.parse.urlsplit(base if "://" in base else "http://" + base).path.rstrip("/")
        self.floored = 0                   # labels read as the floor value (outside the top n_probs)

    def call(self, method, path, body=None):
        if method == "GET" and path == "/v1/models":
            # the shim checks the served model name and context; llama-server serves one model, so report the
            # shim's own name with llama-server's context size
            status, props = super().call("GET", self.prefix + "/props")
            if status != 200 or not isinstance(props, dict):
                return status, props
            n_ctx = props.get("default_generation_settings", {}).get("n_ctx")
            return 200, {"object": "list", "data": [{"id": self.model, "max_model_len": n_ctx}]}
        return super().call(method, self.prefix + path, body)

    def tokenize(self, text):
        # the prompt carries the chat template's special tokens as text, as vLLM's /tokenize parses them
        return self.post("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]

    def detokenize(self, ids):
        return self.post("/detokenize", {"tokens": ids})["content"]

    def label_logprobs(self, prompt, ids):
        """{token id: log-probability} at the answer slot, from one forward pass."""
        r = self.post("/completion", {"prompt": prompt, "n_predict": 1, "n_probs": self.n_probs,
                                      "cache_prompt": True, "temperature": 0.0})
        try:
            top = r["completion_probabilities"][0]["top_logprobs"]
            got = {int(t["id"]): float(t["logprob"]) for t in top}
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise shim.UpstreamError(f"llama-server returned no probabilities at the answer slot ({e})") from e
        floor = min(got.values())
        missing = [i for i in ids if i not in got]
        if missing:
            self.floored += len(missing)
            sys.stderr.write(f"llama_backend: {len(missing)} of {len(ids)} labels outside the top {self.n_probs}, "
                             f"read at the floor {floor:.2f}\n")
        usage = {"prompt_tokens": int(r.get("tokens_evaluated") or 0)}
        return {i: got.get(i, floor) for i in ids}, usage

    def label_logprobs_chat(self, messages, ids, max_pixels=0):
        raise shim.Unprocessable("images are not supported here: the GGUF has no vision projector")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=os.path.join(HERE, "serve_config.json"))
    ap.add_argument("--upstream", default="http://127.0.0.1:8033/upstream/h2o-lightning-4b",
                    help="llama-server base URL (llama-swap's /upstream/<model> path loads the model on demand)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8741)
    ap.add_argument("--n-probs", type=int, default=LlamaServer.n_probs,
                    help="how many top tokens llama-server returns at the answer slot")
    ap.add_argument("--min-context", type=int, default=4096)
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config))
    cfg.pop("image", None)
    LlamaServer.n_probs = a.n_probs
    contract = shim.Contract(cfg)
    s = shim.Shim(contract, LlamaServer(a.upstream, contract.model), a.min_context)
    srv = shim.Server((a.host, a.port), shim.make_handler(s))
    print(f"h2o shim on {a.host}:{a.port} -> llama-server {a.upstream} (model {contract.model}, "
          f"T {contract.temperature}, yes/no floor {contract.noul_floor}, {len(contract.labels)} labels, "
          f"top {a.n_probs})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
