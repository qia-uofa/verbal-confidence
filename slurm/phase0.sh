#!/usr/bin/env bash
# =============================================================================
# Phase 0 — Generate model answers
# Goethe-HLR Frankfurt Cluster (AMD MI210 GPUs)
# =============================================================================
#SBATCH --job-name=vc_phase0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:2
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/logs/verbal-confidence/phase0_%j.out
#SBATCH --error=/scratch/%u/logs/verbal-confidence/phase0_%j.err
#SBATCH --mail-type=FAIL
# Uncomment and set your email:
# #SBATCH --mail-user=you@example.com

set -euo pipefail
source "$(dirname "$0")/env_setup.sh"

# Optional: copy HF cache to local NVMe for faster I/O during the job
LOCAL_CACHE="/local/${SLURM_JOB_ID}/hf_cache"
if [ -d "/local/${SLURM_JOB_ID}" ]; then
    echo "[phase0] Copying HF cache to local NVMe..."
    mkdir -p "${LOCAL_CACHE}"
    rsync -a --ignore-missing-args "${HF_HOME}/" "${LOCAL_CACHE}/" || true
    export HF_HOME="${LOCAL_CACHE}"
    export HF_DATASETS_CACHE="${LOCAL_CACHE}/datasets"
    export TRANSFORMERS_CACHE="${LOCAL_CACHE}/hub"
fi

echo "[phase0] Starting at $(date)"
echo "[phase0] Node: $(hostname), GPUs: ${SLURM_GPUS_ON_NODE:-2}"

python "${PROJECT_ROOT}/scripts/run_phase0.py" \
    --config "${PROJECT_ROOT}/config/default.yaml" \
    --run-name "${SLURM_JOB_ID}" \
    "$@"

echo "[phase0] Done at $(date)"
