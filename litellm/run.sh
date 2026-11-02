#!/bin/bash
# Start the LiteLLM gateway (Opik tracing) in front of llama-swap. Used by litellm.service.
set -euo pipefail
cd "$(dirname "$0")"
python3 gen_config.py
set -a; . ~/.config/litellm/env; set +a
exec ~/lab/litellm/.venv/bin/litellm --config config.generated.yaml --host 127.0.0.1 --port 8034 --num_workers 1
