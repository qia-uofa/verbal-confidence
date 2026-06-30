#!/usr/bin/env bash
# =============================================================================
# env_setup.sh — Sourced by all SLURM job scripts.
# Sets up the Python environment and storage paths on Goethe-HLR.
#
# Storage layout:
#   /home/$USER           30 GB NFS  — source code only
#   /scratch/$GROUP/$USER  5 TB Lustre — ephemeral: models, cache (EPHEMERAL_ROOT)
#   /home/$USER or /work   permanent  — results, logs (PERMANENT_ROOT)
#   /local/$SLURM_JOB_ID  1.4 TB NVMe — temp, deleted after job
# =============================================================================

# ---------- Load .env (only sets variables not already in environment) ----------
DOTENV="${PROJECT_ROOT:-${HOME}/verbal-confidence}/.env"
if [ -f "${DOTENV}" ]; then
    echo "[env_setup] Loading ${DOTENV}"
    set -o allexport
    # shellcheck disable=SC1090
    source "${DOTENV}"
    set +o allexport
fi

# ---------- Validate required variables ----------
if [ -z "${EPHEMERAL_ROOT:-}" ] || [ -z "${PERMANENT_ROOT:-}" ]; then
    echo "ERROR: EPHEMERAL_ROOT and PERMANENT_ROOT must be set in .env or the shell."
    echo "       Copy .env.example to .env and fill in your paths."
    exit 1
fi

# ---------- Derive cache/results paths from the two roots ----------
export HF_HOME="${EPHEMERAL_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM="false"

export RESULTS_ROOT="${PERMANENT_ROOT}/results/verbal-confidence"
export LOGS_ROOT="${PERMANENT_ROOT}/logs/verbal-confidence"

# Propagate HF token if set
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

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
