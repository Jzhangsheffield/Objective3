#!/bin/bash
#SBATCH --job-name=gh_atom_ar
#SBATCH --array=1-9
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
  1) MODEL_NAME="m3_atomic_tail_frozen_m0_delta"; REFRESH="1"; LABEL="refresh_every_1" ;;
  2) MODEL_NAME="m3_atomic_tail_joint_head_delta"; REFRESH="1"; LABEL="refresh_every_1" ;;
  3) MODEL_NAME="m3_atomic_tail_direct_fusion"; REFRESH="1"; LABEL="refresh_every_1" ;;
  4) MODEL_NAME="m3_atomic_tail_frozen_m0_delta"; REFRESH="10"; LABEL="refresh_every_10" ;;
  5) MODEL_NAME="m3_atomic_tail_joint_head_delta"; REFRESH="10"; LABEL="refresh_every_10" ;;
  6) MODEL_NAME="m3_atomic_tail_direct_fusion"; REFRESH="10"; LABEL="refresh_every_10" ;;
  7) MODEL_NAME="m3_atomic_tail_frozen_m0_delta"; REFRESH="once"; LABEL="refresh_once" ;;
  8) MODEL_NAME="m3_atomic_tail_joint_head_delta"; REFRESH="once"; LABEL="refresh_once" ;;
  9) MODEL_NAME="m3_atomic_tail_direct_fusion"; REFRESH="once"; LABEL="refresh_once" ;;
  *) echo "Unsupported array task: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

if [[ ! -f "${ALLRUN_FEATURE_ROOT}/completed.json" ]]; then
  if [[ ! -f "${ALLRUN_FEATURE_ROOT}/train_all.pt" || ! -f "${ALLRUN_FEATURE_ROOT}/test_all.pt" ]]; then
    echo "Incomplete all-runs Tier-3 feature cache: ${ALLRUN_FEATURE_ROOT}" >&2
    exit 1
  fi
  echo "Legacy all-runs feature cache has no completed.json; verified cache files."
fi
M0_CHECKPOINT="${ALLRUN_MODEL_ROOT}/all_runs/m0/last.pth"
if [[ "${MODEL_NAME}" == "m3_atomic_tail_frozen_m0_delta" && ! -f "${M0_CHECKPOINT}" ]]; then
  echo "Missing all-runs M0 checkpoint: ${M0_CHECKPOINT}" >&2
  exit 1
fi
if [[ -f "${ATOMIC_TAIL_MODEL_ROOT}/all_runs/${LABEL}/${MODEL_NAME}/completed.json" ]]; then
  echo "Completed all-runs ${MODEL_NAME} ${LABEL} exists; skip."
  exit 0
fi

extra_args=()
if [[ "${MODEL_NAME}" == "m3_atomic_tail_frozen_m0_delta" ]]; then
  extra_args+=(--m0-checkpoint "${M0_CHECKPOINT}")
fi
"${PYTHON_BIN}" tools/train_atomic_tail_graph_valid.py \
  --model "${MODEL_NAME}" \
  --train-scope all_runs \
  --protocol-root "${PROTOCOL_ROOT}" \
  --train-cache "${ALLRUN_FEATURE_ROOT}/train_all.pt" \
  --test-cache "${ALLRUN_FEATURE_ROOT}/test_all.pt" \
  --task-graph "${TASK_GRAPH}" \
  --relation-matrix "${RELATION_MATRIX}" \
  --output-root "${ATOMIC_TAIL_MODEL_ROOT}" \
  --shuffle-refresh-interval "${REFRESH}" \
  --epochs "${HISTORY_EPOCHS}" \
  --batch-size 64 \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  "${extra_args[@]}"
