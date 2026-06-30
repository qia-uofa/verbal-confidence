#!/usr/bin/env bash
# =============================================================================
# submit_phase0.sh — Submit phase0.sh with log paths from .env
#
# Usage:
#   bash slurm/submit_phase0.sh [extra sbatch args]
#
# Sources .env to resolve PERMANENT_ROOT, then passes --output/--error
# so SLURM logs land in PERMANENT_ROOT (not a hardcoded path).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------- Load .env ----------
DOTENV="${PROJECT_ROOT}/.env"
if [ ! -f "${DOTENV}" ]; then
    echo "ERROR: .env not found at ${DOTENV}"
    echo "       Copy .env.example to .env and fill in your paths."
    exit 1
fi
set -o allexport
# shellcheck disable=SC1090
source "${DOTENV}"
set +o allexport

# ---------- Validate ----------
if [ -z "${EPHEMERAL_ROOT:-}" ] || [ -z "${PERMANENT_ROOT:-}" ]; then
    echo "ERROR: EPHEMERAL_ROOT and PERMANENT_ROOT must be set in .env"
    exit 1
fi

# ---------- SLURM account / partition ----------
ACCOUNT_ARG=()
[[ -n "${SLURM_ACCOUNT:-}" ]] && ACCOUNT_ARG=(--account="${SLURM_ACCOUNT}")
PARTITION="${SLURM_PARTITION:-gpu}"

# ---------- Prepare log directory ----------
LOG_DIR="${PERMANENT_ROOT}/logs/verbal-confidence"
mkdir -p "${LOG_DIR}"

# ---------- Submit ----------
echo "Submitting phase0.sh"
echo "  EPHEMERAL_ROOT = ${EPHEMERAL_ROOT}"
echo "  PERMANENT_ROOT = ${PERMANENT_ROOT}"
echo "  Logs           = ${LOG_DIR}"
echo "  Partition      = ${PARTITION}"
[[ -n "${SLURM_ACCOUNT:-}" ]] && echo "  Account        = ${SLURM_ACCOUNT}"

JOB_ID=$(sbatch --parsable \
    --partition="${PARTITION}" \
    "${ACCOUNT_ARG[@]}" \
    --output="${LOG_DIR}/phase0_%j.out" \
    --error="${LOG_DIR}/phase0_%j.err" \
    "${SCRIPT_DIR}/phase0.sh" \
    "$@")

echo "  Job ID: ${JOB_ID}"
echo "  stdout: ${LOG_DIR}/phase0_${JOB_ID}.out"
echo "  stderr: ${LOG_DIR}/phase0_${JOB_ID}.err"
