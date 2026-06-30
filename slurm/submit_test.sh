#!/usr/bin/env bash
# =============================================================================
# submit_test.sh — Submit test_run.sh with log paths from .env
#
# Usage:
#   bash slurm/submit_test.sh [extra sbatch args]
#
# Sources .env to resolve PERMANENT_ROOT, then passes --output/--error
# so SLURM logs land in PERMANENT_ROOT (not a hardcoded path).
# =============================================================================
# Runs on the 'test' partition (<2h) — useful for verifying setup.

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

# ---------- SLURM partition ----------
# Set SLURM_TEST_PARTITION in .env to match your cluster (check with: sinfo -s).
PARTITION="${SLURM_TEST_PARTITION:-test}"

# ---------- Prepare log directory ----------
LOG_DIR="${PERMANENT_ROOT}/logs/verbal-confidence"
mkdir -p "${LOG_DIR}"

# ---------- Submit ----------
echo "Submitting test_run.sh"
echo "  EPHEMERAL_ROOT = ${EPHEMERAL_ROOT}"
echo "  PERMANENT_ROOT = ${PERMANENT_ROOT}"
echo "  Logs           = ${LOG_DIR}"
echo "  Partition      = ${PARTITION}"

JOB_ID=$(sbatch --parsable \
    --partition="${PARTITION}" \
    --output="${LOG_DIR}/test_%j.out" \
    --error="${LOG_DIR}/test_%j.err" \
    "${SCRIPT_DIR}/test_run.sh" \
    "$@")

echo "  Job ID: ${JOB_ID}"
echo "  stdout: ${LOG_DIR}/test_${JOB_ID}.out"
echo "  stderr: ${LOG_DIR}/test_${JOB_ID}.err"
