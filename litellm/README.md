# LiteLLM gateway: llama-swap with Opik tracing

An OpenAI-compatible gateway on **port 8034** that forwards every request to llama-swap
(`127.0.0.1:8033`) and logs it to [Opik](http://ssd-nodes.akita-betelgeuse.ts.net:5173)
on the ssd-nodes VPS. llama-swap itself is unchanged: only agents pointed at 8034 are traced,
so benchmarks and scripts that use 8033 stay out of Opik.

| | Address |
|---|---|
| On Marvin | `http://127.0.0.1:8034/v1` |
| Tailnet (once `tailscale serve` is set up, below) | `https://marvin.akita-betelgeuse.ts.net:8034/v1` |
| Not traced (llama-swap directly) | `…:8033/v1`, as before |

## Choosing the Opik project: the API key

The API key an agent sends picks its Opik project. `opik-<project>` logs to `<project>`,
so `opik-hermes` logs to `hermes`. Any other key, or none, is accepted and logs to
`unlabelled`. The keys aren't secret: the gateway is tailnet-only, like llama-swap.

## What gets logged

- Chat completions, streamed or not, with tool calls, reasoning and token counts. Audio
  and the rest are logged too.
- **Not embeddings** (`tracing.py`). They are the input text plus a vector, which is useless
  for evals, and an indexer re-embedding a vault would flood a project with them.
- Extra parameters (`chat_template_kwargs`, `top_k`, ...) are passed through to llama-swap
  unchanged.
- Overhead: none measurable (median ~0.04 s per 1-token request either way, 2026-11-02).

## Files

| File | What |
|---|---|
| `gen_config.py` | Writes `config.generated.yaml`. Every llama-swap model and alias becomes a route, plus a `*` route that passes any other name through. |
| `auth.py` | Maps the API key to an Opik project; uses LiteLLM's custom auth, so no key database is needed. |
| `tracing.py` | LiteLLM's Opik logger, minus embeddings. |
| `run.sh` | Regenerates the config, then starts LiteLLM. |
| `litellm.service` | User service that runs `run.sh`. |
| `litellm-reload.path`, `litellm-reload.service` | Restart the gateway when `../config.yaml` changes, so its model list follows llama-swap's. |

Elsewhere:

- `~/lab/litellm/.venv`: LiteLLM 1.106.6 (Python 3.12). Rebuild it with
  `uv venv ~/lab/litellm/.venv --python 3.12 && VIRTUAL_ENV=~/lab/litellm/.venv uv pip install 'litellm[proxy]'`.
- `~/.config/litellm/env` (mode 600, not in git): `LITELLM_MASTER_KEY` (random; LiteLLM
  refuses to start without one), the Opik URL, workspace `default`, and the default project
  `unlabelled`.

## Install

```bash
for u in litellm.service litellm-reload.path litellm-reload.service; do
  ln -sf ~/llama-swap/litellm/$u ~/.config/systemd/user/$u; done
systemctl --user daemon-reload
systemctl --user enable --now litellm.service litellm-reload.path
tailscale serve --bg --https=8034 http://127.0.0.1:8034   # tailnet HTTPS (may need sudo)
```

Logs: `journalctl --user -u litellm -f`.
