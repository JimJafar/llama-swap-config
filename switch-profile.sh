#!/usr/bin/env bash
set -euo pipefail

PROFILE=${1:-}
NEED_GEMMA=1
case "$PROFILE" in
27b) WORKLOAD=Q3.8-27B-IQ4XS ;;
dflash2) WORKLOAD=Q3.8-27B-IQ4XS-DF2 ;;
q6) WORKLOAD=Q3.8-27B-Q6KM ;;
flash)
	# GSQ-RCO Q2_0 (ISTA) rather than the AtomicChat IQ4_XS build: switched 2026-09-17.
	# Same three-card set as the AtomicChat entry (5070 Ti + 5060 Ti + 3090) and the same
	# swap group, so the group logic below is unchanged. Bonus: -c 262144 vs 196608.
	WORKLOAD=Q3.8-FN-GSQ-RCO-Q2_0
	NEED_GEMMA=1
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
FACTORY_MODEL=Q3.8-27B-IQ4XS

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

# Every profile now keeps Gemma resident. The 27B profiles use the MSI + 5070 pair and
# the flash workloads use the 5070 Ti + 5060 Ti + 3090 trio -- all of them EXCLUDE the
# Zotac (af5bd53f), which is where Gemma lives, so none of them contend for its card.
# The old rule skipped Gemma for the flash profile on the grounds that "Flash fills all
# three cards"; that dated from when Flash still held the Zotac, before the 2026-09-12
# change that put the Flash-Next --fit entries back to 3 GPUs. Flipped 2026-09-17.
# Loading the WORKLOAD first is still required: its request evicts the previous
# swap-group member and frees those cards before Gemma is requested.
if [ "$NEED_GEMMA" = 1 ]; then
	echo "Loading resident Gemma..."
	request gemma-4-E4B-MTP
fi
echo "Active profile: $PROFILE"
