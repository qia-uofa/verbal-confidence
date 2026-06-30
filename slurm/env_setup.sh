#!/usr/bin/env bash
# =============================================================================
# env_setup.sh — Sourced by all SLURM job scripts.
# Sets up the Python environment and HuggingFace cache paths on Goethe-HLR.
# =============================================================================

# ---------- Storage layout ----------
# /home/$USER          : 30 GB NFS (code lives here)
# /scratch/$GROUP/$USER: 5 TB Lustre (models, datasets, results, logs)
# /local/$SLURM_JOB_ID : 1.4 TB NVMe (temporary, deleted after job)

export GROUP="${GROUP:-$(id -gn)}"
export SCRATCH="/scratch/${GROUP}/${USER}"

# HuggingFace cache → fast scratch
export HF_HOME="${SCRATCH}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM="false"

# Results
export RESULTS_ROOT="${SCRATCH}/results/verbal-confidence"
export LOGS_ROOT="${SCRATCH}/logs/verbal-confidence"

# Make required directories
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" \
         "${RESULTS_ROOT}" "${LOGS_ROOT}"

# ---------- ROCm / MI210 ----------
# The HLR GPU nodes use AMD MI210 with ROCm.
# Set HSA_OVERRIDE_GFX_VERSION if your ROCm version doesn't auto-detect gfx90a.
# export HSA_OVERRIDE_GFX_VERSION=9.0.a

# ---------- Python ----------
# Activate your conda/venv environment.
# Adjust the path to your actual environment:
ENV_PATH="${HOME}/envs/verbal-confidence"
if [ -f "${ENV_PATH}/bin/activate" ]; then
    source "${ENV_PATH}/bin/activate"
elif command -v conda &>/dev/null; then
    conda activate verbal-confidence 2>/dev/null || true
fi

# ---------- Project root ----------
export PROJECT_ROOT="${HOME}/verbal-confidence"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

echo "[env_setup] SCRATCH=${SCRATCH}"
echo "[env_setup] HF_HOME=${HF_HOME}"
echo "[env_setup] PYTHONPATH=${PYTHONPATH}"
