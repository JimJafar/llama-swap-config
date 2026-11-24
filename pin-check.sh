# pin-check.sh (sourced, 2026-11-24): refuse to start while another model holds the pin.
# See pin.sh. The pinned model's own cmd runs under `pin.sh on`, which sets PIN_OWNER.
if [ -s /home/jim/llama-swap/pinned ] && [ "$(cat /home/jim/llama-swap/pinned)" != "${PIN_OWNER:-}" ]; then
    echo "pinned: $(cat /home/jim/llama-swap/pinned) holds the GPUs; not starting." \
         "Lift the pin with ~/llama-swap/pin.sh off" >&2
    exit 1
fi
