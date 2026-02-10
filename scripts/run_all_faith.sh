#!/usr/bin/env bash
# Run faithfulness rollouts for all 41 questions (uncued + cued).
#
# Usage:
#   bash scripts/run_all_faith.sh              # default: rollouts
#   bash scripts/run_all_faith.sh flowcharts   # or any other command
#
# Requires: generate_faith_configs.py to have been run first.

set -euo pipefail

COMMAND="${1:-rollouts}"
MAX_WORKERS="${MAX_WORKERS:-20}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-2}"

# All 41 pn values from the good_problems dataset
PNS=(19 26 37 59 60 62 68 81 119 145 198 212 215 277 288 295 309 324 339 340 369 371 382 408 418 430 432 440 459 474 479 499 700 715 768 804 819 827 877 960 972)

TOTAL=${#PNS[@]}
COUNT=0

echo "Running ${COMMAND} for ${TOTAL} questions (${TOTAL}×2 = $((TOTAL * 2)) configs)"
echo "=========================================="

for pn in "${PNS[@]}"; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "[${COUNT}/${TOTAL}] pn=${pn} — uncued"
    python -m src.main --config-name="faith_uncued_pn${pn}" "command=${COMMAND}" "r.max_workers=${MAX_WORKERS}"

    sleep "${SLEEP_BETWEEN}"

    echo "[${COUNT}/${TOTAL}] pn=${pn} — cued"
    python -m src.main --config-name="faith_cued_pn${pn}" "command=${COMMAND}" "r.max_workers=${MAX_WORKERS}"

    sleep "${SLEEP_BETWEEN}"
done

echo ""
echo "=========================================="
echo "Done: ${COMMAND} for all ${TOTAL} questions."
