#!/usr/bin/env bash
# =============================================================================
# test_run.sh — Quick smoke test on the "test" partition (<2h limit).
# Runs only Phase 0+1 on 20 questions to verify the setup.
# =============================================================================
#SBATCH --job-name=vc_test
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:2
#SBATCH --mem=0
#SBATCH --time=01:30:00
# Note: --output/--error are intentionally absent. #SBATCH cannot read shell
# variables or .env. Use submit_all.sh, which sources .env and passes
# --output/--error from PERMANENT_ROOT. Direct sbatch logs to slurm-JOBID.out.

set -euo pipefail
source "$(dirname "$0")/env_setup.sh"

echo "[test] Starting smoke test at $(date)"

# Write a minimal override config for quick testing
TEST_CONFIG="/tmp/vc_test_${SLURM_JOB_ID}.yaml"
cat > "${TEST_CONFIG}" << EOF
n_questions: 20
seed: 0
paths:
  run_name: test_${SLURM_JOB_ID}
EOF

python "${PROJECT_ROOT}/scripts/run_all_experiments.py" \
    --config "${TEST_CONFIG}" \
    --run-name "test_${SLURM_JOB_ID}" \
    --skip generalization variance_partitioning

rm -f "${TEST_CONFIG}"
echo "[test] Smoke test done at $(date)"
