#!/usr/bin/env bash
# =============================================================================
# Phase 1 — Confidence elicitation
# =============================================================================
#SBATCH --job-name=vc_phase1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:2
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/logs/verbal-confidence/phase1_%j.out
#SBATCH --error=/scratch/%u/logs/verbal-confidence/phase1_%j.err
#SBATCH --mail-type=FAIL
# #SBATCH --mail-user=you@example.com

set -euo pipefail
source "$(dirname "$0")/env_setup.sh"

echo "[phase1] Starting at $(date)"

python "${PROJECT_ROOT}/scripts/run_phase1.py" \
    --config "${PROJECT_ROOT}/config/default.yaml" \
    --run-name "${SLURM_JOB_ID}" \
    "$@"

echo "[phase1] Done at $(date)"
