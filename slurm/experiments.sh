#!/usr/bin/env bash
# =============================================================================
# Full experiment pipeline (Exps 1–7 + generalization)
# Long job — requests full node for up to 24 h.
# =============================================================================
#SBATCH --job-name=vc_experiments
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:2
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/logs/verbal-confidence/experiments_%j.out
#SBATCH --error=/home/%u/logs/verbal-confidence/experiments_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=you@example.com

set -euo pipefail
source "$(dirname "$0")/env_setup.sh"

# Optional: stage HF cache to local NVMe
LOCAL_CACHE="/local/${SLURM_JOB_ID}/hf_cache"
if [ -d "/local/${SLURM_JOB_ID}" ]; then
    echo "[experiments] Staging HF cache to ${LOCAL_CACHE}..."
    mkdir -p "${LOCAL_CACHE}"
    rsync -a --ignore-missing-args "${HF_HOME}/" "${LOCAL_CACHE}/" || true
    export HF_HOME="${LOCAL_CACHE}"
    export HF_DATASETS_CACHE="${LOCAL_CACHE}/datasets"
    export TRANSFORMERS_CACHE="${LOCAL_CACHE}/hub"
fi

echo "[experiments] Starting at $(date)"
echo "[experiments] Node: $(hostname)"
echo "[experiments] GPUs: $(rocm-smi --showproductname 2>/dev/null | head -4 || echo 'N/A')"

# Run name = SLURM job ID for traceability
RUN_NAME="${SLURM_JOB_ID}"

python "${PROJECT_ROOT}/scripts/run_all_experiments.py" \
    --config "${PROJECT_ROOT}/config/default.yaml" \
    --run-name "${RUN_NAME}" \
    "$@"

echo "[experiments] Done at $(date)"
echo "[experiments] Results: ${RESULTS_ROOT}/${RUN_NAME}/"
