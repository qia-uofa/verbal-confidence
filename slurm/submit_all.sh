#!/usr/bin/env bash
# =============================================================================
# submit_all.sh — Submit the full pipeline as a SLURM job dependency chain.
#
# Usage:
#   bash slurm/submit_all.sh [--model qwen] [--config config/custom.yaml]
#
# Jobs submitted:
#   1. phase0  → generates answers
#   2. phase1  → confidence elicitation (depends on phase0)
#   3. experiments → all 7 experiments (depends on phase1)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL=""
CONFIG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2";  shift 2 ;;
        --config)  CONFIG="$2"; shift 2 ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

EXTRA_ARGS=""
[[ -n "$MODEL"  ]] && EXTRA_ARGS="${EXTRA_ARGS} --model ${MODEL}"
[[ -n "$CONFIG" ]] && EXTRA_ARGS="${EXTRA_ARGS} --config ${CONFIG}"

echo "=== Submitting Verbal Confidence pipeline ==="

JOB0=$(sbatch --parsable "${SCRIPT_DIR}/phase0.sh" ${EXTRA_ARGS})
echo "  Phase 0 job ID: ${JOB0}"

JOB1=$(sbatch --parsable --dependency=afterok:${JOB0} \
    "${SCRIPT_DIR}/phase1.sh" ${EXTRA_ARGS})
echo "  Phase 1 job ID: ${JOB1}"

JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} \
    "${SCRIPT_DIR}/experiments.sh" ${EXTRA_ARGS})
echo "  Experiments job ID: ${JOB2}"

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Cancel all:    scancel ${JOB0} ${JOB1} ${JOB2}"
