#!/bin/bash
#SBATCH --job-name=gh_direct_no
#SBATCH --array=1-3
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/parscratch/users/mes19jz/objective3/obj2_codes/ft_and_test/out/slurm-%x-%A_%a.out
#SBATCH --error=/mnt/parscratch/users/mes19jz/objective3/obj2_codes/ft_and_test/out/slurm-%x-%A_%a.err
set -euo pipefail
PACKAGE_ROOT="${PACKAGE_ROOT:-/mnt/parscratch/users/mes19jz/objective3/codex_and_files/graph_history_rgb_cross_person_ADM_2026-07-22}"
source "${PACKAGE_ROOT}/slurm/config_hpc.sh"
setup_hpc_environment
MODEL_NAME="m${SLURM_ARRAY_TASK_ID}_direct"
if [[ ! -f "${FEATURE_ROOT}/completed.json" ]]; then
  if [[ ! -f "${FEATURE_ROOT}/train_all.pt" || ! -f "${FEATURE_ROOT}/test_all.pt" ]]; then
    echo "Incomplete normal-only Tier-3 feature cache: ${FEATURE_ROOT}" >&2
    exit 1
  fi
  echo "Legacy normal-only feature cache has no completed.json; verified train_all.pt and test_all.pt."
fi
if [[ -f "${DIRECT_MODEL_ROOT}/normal_only/${MODEL_NAME}/completed.json" ]]; then
  echo "Completed normal-only ${MODEL_NAME} exists; skip."
  exit 0
fi
"${PYTHON_BIN}" tools/train_direct_history_model.py \
  --model "${MODEL_NAME}" --train-scope normal_only --protocol-root "${PROTOCOL_ROOT}" \
  --train-cache "${FEATURE_ROOT}/train_all.pt" --test-cache "${FEATURE_ROOT}/test_all.pt" \
  --task-graph "${TASK_GRAPH}" --relation-matrix "${RELATION_MATRIX}" \
  --output-root "${DIRECT_MODEL_ROOT}" --epochs "${HISTORY_EPOCHS}" --batch-size 64 \
  --num-workers "${NUM_WORKERS}" --seed "${SEED}"
