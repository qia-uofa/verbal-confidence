#!/usr/bin/env bash
# =============================================================================
# submit_all.sh — Submit the full pipeline as a SLURM dependency chain.
#
# This script sources .env so PERMANENT_ROOT is known at submission time,
# then passes --output / --error to each sbatch call so logs land in the
# right place (not hardcoded /scratch or /home).
#
# #SBATCH --output lines inside the job scripts are only a fallback for
# direct `sbatch` calls; this wrapper always overrides them.
#
# Usage:
#   bash slurm/submit_all.sh [--model qwen] [--config config/custom.yaml]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------- Load .env (only sets vars not already in shell) ----------
DOTENV="${PROJECT_ROOT}/.env"
if [ -f "${DOTENV}" ]; then
    set -o allexport
    # shellcheck disable=SC1090
    source "${DOTENV}"
    set +o allexport
fi

if [ -z "${EPHEMERAL_ROOT:-}" ] || [ -z "${PERMANENT_ROOT:-}" ]; then
    echo "ERROR: EPHEMERAL_ROOT and PERMANENT_ROOT must be set in .env or the shell."
    exit 1
fi

# ---------- SLURM partition ----------
PARTITION="${SLURM_PARTITION:-gpu}"

# Permanent log directory (created now so SLURM can write to it immediately)
LOG_DIR="${PERMANENT_ROOT}/logs/verbal-confidence"
mkdir -p "${LOG_DIR}"

# ---------- Parse args ----------
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
echo "    EPHEMERAL_ROOT = ${EPHEMERAL_ROOT}"
echo "    PERMANENT_ROOT = ${PERMANENT_ROOT}"
echo "    Logs           = ${LOG_DIR}"
echo "    Partition      = ${PARTITION}"
echo ""

# ---------- Submit with dependency chain ----------
JOB0=$(sbatch --parsable \
    --partition="${PARTITION}" \
    --output="${LOG_DIR}/phase0_%j.out" \
    --error="${LOG_DIR}/phase0_%j.err" \
    "${SCRIPT_DIR}/phase0.sh" ${EXTRA_ARGS})
echo "  Phase 0 job ID: ${JOB0}"

JOB1=$(sbatch --parsable \
    --partition="${PARTITION}" \
    --dependency=afterok:${JOB0} \
    --output="${LOG_DIR}/phase1_%j.out" \
    --error="${LOG_DIR}/phase1_%j.err" \
    "${SCRIPT_DIR}/phase1.sh" ${EXTRA_ARGS})
echo "  Phase 1 job ID: ${JOB1}"

JOB2=$(sbatch --parsable \
    --partition="${PARTITION}" \
    --dependency=afterok:${JOB1} \
    --output="${LOG_DIR}/experiments_%j.out" \
    --error="${LOG_DIR}/experiments_%j.err" \
    "${SCRIPT_DIR}/experiments.sh" ${EXTRA_ARGS})
echo "  Experiments job ID: ${JOB2}"

echo ""
echo "Monitor:    squeue -u \$USER"
echo "Logs:       ${LOG_DIR}/"
echo "Cancel all: scancel ${JOB0} ${JOB1} ${JOB2}"
