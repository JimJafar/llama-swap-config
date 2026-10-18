#!/bin/sh
# series.sh LABEL LEVEL REPS [--append FILE]: every task REPS times, interleaved (calc, ledger, sheet, calc, ...)
set -u
LABEL=$1; LEVEL=$2; REPS=$3; shift 3
cd "$(dirname "$0")"
i=1
while [ "$i" -le "$REPS" ]; do
    for t in calc ledger sheet; do
        python3 run.py "$t" "$LEVEL" "$LABEL" "$@"
    done
    i=$((i + 1))
done
