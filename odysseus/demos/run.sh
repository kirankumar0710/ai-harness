#!/usr/bin/env bash
# Run the demos and save each run's complete output to its own file.
#
#   demos/run.sh                  every demo, offline and live (incl. gate --ask)
#   demos/run.sh offline          every demo with --offline (no API key needed)
#   demos/run.sh live             every demo against the real model
#   demos/run.sh live gate memory only these demos, live
#
# Files go to demos/runs/<timestamp>/, named <demo>-<mode>_<timestamp>.txt:
#
#   dice-offline_20261116-153012.txt
#   gate-live-ask_20261116-153012.txt
#
# so each one says which demo and mode produced it. demos/runs/ is in
# .gitignore. Paths are cleaned (home -> ~, temp -> <tmp>) by record.py.
#
# Live runs need ODYSSEUS_PROVIDER and that provider's API key (see the
# dice walkthrough). gate --ask stops for y/N answers in scene 2.

set -u

DEMOS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$DEMOS_DIR")"
PYTHON="${PYTHON:-python3}"
ALL_DEMOS="dice gate skills memory context durability fleet"

# Run from a checkout even without "pip install -e .".
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

usage() {
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
}

MODE="${1:-all}"
case "$MODE" in
    all|offline|live) shift || true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown mode: $MODE (use all, offline or live)" >&2
       usage; exit 2 ;;
esac
DEMOS="${*:-$ALL_DEMOS}"
for demo in $DEMOS; do
    if [ ! -f "$DEMOS_DIR/$demo/$demo.py" ]; then
        echo "unknown demo: $demo (choose from: $ALL_DEMOS)" >&2
        exit 2
    fi
done

if [ "$MODE" != "offline" ] && [ -z "${ODYSSEUS_PROVIDER:-}" ]; then
    echo "note: ODYSSEUS_PROVIDER is not set, so the live runs will fail" \
         "with a message saying what to set." >&2
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$DEMOS_DIR/runs/$STAMP"
mkdir -p "$RUN_DIR"
RESULTS=""

# run <demo> <mode label> [demo options...]
run() {
    local demo="$1" label="$2"
    shift 2
    local file="$RUN_DIR/$demo-${label}_$STAMP.txt"
    echo
    echo "################ $demo ($label) ################"
    "$PYTHON" "$DEMOS_DIR/record.py" "$demo" --out "$file" "$@"
    local code=$?
    RESULTS="$RESULTS$(printf '  exit %-3s %s' "$code" "${file#$PROJECT_DIR/}")
"
}

if [ "$MODE" = "all" ] || [ "$MODE" = "offline" ]; then
    for demo in $DEMOS; do
        run "$demo" offline --offline
    done
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "live" ]; then
    for demo in $DEMOS; do
        run "$demo" live
        if [ "$demo" = "gate" ]; then
            run gate live-ask --ask
        fi
    done
fi

echo
echo "################ summary ################"
printf '%s' "$RESULTS"
echo "folder: ${RUN_DIR#$PROJECT_DIR/}"
