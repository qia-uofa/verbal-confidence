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
#SBATCH --mail-type=FAIL
# Note: --output/--error are intentionally absent. #SBATCH cannot read shell
# variables or .env. Use submit_all.sh, which sources .env and passes
# --output/--error from PERMANENT_ROOT. Direct sbatch logs to slurm-JOBID.out.
# #SBATCH --mail-user=you@example.com

set -euo pipefail
source "${PROJECT_ROOT}/slurm/env_setup.sh"

echo "[phase1] Starting at $(date)"

python "${PROJECT_ROOT}/scripts/run_phase1.py" \
    --config "${PROJECT_ROOT}/config/default.yaml" \
    --run-name "${SLURM_JOB_ID}" \
    "$@"

echo "[phase1] Done at $(date)"
