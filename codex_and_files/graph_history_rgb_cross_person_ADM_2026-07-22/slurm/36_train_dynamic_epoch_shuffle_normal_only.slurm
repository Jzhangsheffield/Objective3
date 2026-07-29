#!/bin/bash
#SBATCH --job-name=gh_dyn_no
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

case "${SLURM_ARRAY_TASK_ID}" in
  1) MODEL_NAME="m3_dynamic_frozen_m0_delta" ;;
  2) MODEL_NAME="m3_dynamic_joint_head_delta" ;;
  3) MODEL_NAME="m3_dynamic_direct_fusion" ;;
  *) echo "Unsupported array task: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

if [[ ! -f "${FEATURE_ROOT}/completed.json" ]]; then
  if [[ ! -f "${FEATURE_ROOT}/train_all.pt" || ! -f "${FEATURE_ROOT}/test_all.pt" ]]; then
    echo "Incomplete normal-only Tier-3 feature cache: ${FEATURE_ROOT}" >&2
    exit 1
  fi
  echo "Legacy normal-only feature cache has no completed.json; verified cache files."
fi
M0_CHECKPOINT="${MODEL_ROOT}/normal_only/m0/last.pth"
if [[ "${MODEL_NAME}" == "m3_dynamic_frozen_m0_delta" && ! -f "${M0_CHECKPOINT}" ]]; then
  echo "Missing normal-only M0 checkpoint: ${M0_CHECKPOINT}" >&2
  exit 1
fi
if [[ -f "${DYNAMIC_MODEL_ROOT}/normal_only/${MODEL_NAME}/completed.json" ]]; then
  echo "Completed normal-only ${MODEL_NAME} exists; skip without overwrite."
  exit 0
fi

extra_args=()
if [[ "${MODEL_NAME}" == "m3_dynamic_frozen_m0_delta" ]]; then
  extra_args+=(--m0-checkpoint "${M0_CHECKPOINT}")
fi
"${PYTHON_BIN}" tools/train_dynamic_epoch_shuffle.py \
  --model "${MODEL_NAME}" \
  --train-scope normal_only \
  --protocol-root "${PROTOCOL_ROOT}" \
  --train-cache "${FEATURE_ROOT}/train_all.pt" \
  --test-cache "${FEATURE_ROOT}/test_all.pt" \
  --task-graph "${TASK_GRAPH}" \
  --relation-matrix "${RELATION_MATRIX}" \
  --output-root "${DYNAMIC_MODEL_ROOT}" \
  --epochs "${HISTORY_EPOCHS}" \
  --batch-size 64 \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  "${extra_args[@]}"
