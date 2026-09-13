#!/usr/bin/env bash
set -euo pipefail

PROFILE=${1:-}
NEED_GEMMA=1
case "$PROFILE" in
27b) WORKLOAD=Qwen3.8-27B-UD-IQ4_XS-MTP-TP ;;
dflash2) WORKLOAD=Qwen3.8-27B-UD-IQ4_XS-DFlash2-TP ;;
q6) WORKLOAD=Qwen3.8-27B-UD-Q6_K_M ;;
flash)
	WORKLOAD=Qwen3.8-Flash-Next-UD-IQ4_XS
	NEED_GEMMA=0
	;;
*)
	echo "Usage: $0 {27b|dflash2|q6|flash}" >&2
	exit 64
	;;
esac

BASE_URL=${LLAMA_SWAP_URL:-http://127.0.0.1:8033}

# The sdlc-factory's model is PERSISTENT (group `factory-resident`) so it cannot be
# swapped out from under a running Task -- a cold load inside the factory's bounded
# provider turn is what kept parking its Tasks on an infrastructure Hold.
FACTORY_MODEL=Qwen3.8-27B-UD-IQ4_XS-MTP-TP

request() {
	curl --fail-with-body --silent --show-error --max-time 900 \
		-H 'content-type: application/json' \
		-d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply only: ready\"}],\"max_tokens\":2}" \
		"$BASE_URL/v1/chat/completions" >/dev/null
}

# Load the WORKLOAD first. It shares a swap group with whichever Qwen profile is
# currently resident, so requesting it evicts the previous one (e.g. Flash, which
# fills all three cards) and frees its GPUs. Loading Gemma first failed when Flash
# was still holding the Zotac that Gemma needs (curl 22 / upstream error).
#
# Because the factory model is now PERSISTENT, membership no longer evicts it, so
# Flash-Next (which wants all three compute cards) would fail to load while it is
# resident. Release it explicitly first -- unless the profile we are switching TO
# is that model, in which case unloading it would be pointless churn. The unload is
# a no-op when the model is not resident.
if [ "$WORKLOAD" != "$FACTORY_MODEL" ]; then
	echo "Releasing the persistent factory model..."
	curl --silent --show-error --max-time 900 -X POST \
		"$BASE_URL/api/models/unload/$FACTORY_MODEL" >/dev/null || true
fi
echo "Loading $WORKLOAD..."
request "$WORKLOAD"

# The 27B profiles (27b, dflash2) use only the MSI + 5070 pair, so after the
# workload loads, the Zotac is free and Gemma can be resident there. Flash fills
# all three cards and cannot coexist with Gemma, so skip it for the flash profile.
if [ "$NEED_GEMMA" = 1 ]; then
	echo "Loading resident Gemma..."
	request gemma-4-E4B-MTP
fi
echo "Active profile: $PROFILE"
