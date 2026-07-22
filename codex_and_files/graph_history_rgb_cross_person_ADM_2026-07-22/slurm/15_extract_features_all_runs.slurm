#!/bin/bash
#SBATCH --job-name=gh_ar_features
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/parscratch/users/mes19jz/objective3/obj2_codes/ft_and_test/out/slurm-%x-%j.out
#SBATCH --error=/mnt/parscratch/users/mes19jz/objective3/obj2_codes/ft_and_test/out/slurm-%x-%j.err
set -euo pipefail
PACKAGE_ROOT="${PACKAGE_ROOT:-/mnt/parscratch/users/mes19jz/objective3/codex_and_files/graph_history_rgb_cross_person_ADM_2026-07-22}"
source "${PACKAGE_ROOT}/slurm/config_hpc.sh"
setup_hpc_environment
if [[ -f "${ALLRUN_FEATURE_ROOT}/completed.json" ]]; then
  echo "Completed all-runs feature cache exists; skip: ${ALLRUN_FEATURE_ROOT}"
  exit 0
fi
[[ -f "${ALLRUN_BACKBONE_CKPT}" ]] || { echo "Missing ${ALLRUN_BACKBONE_CKPT}" >&2; exit 1; }
"${PYTHON_BIN}" tools/guard_output_dir.py --output-dir "${ALLRUN_FEATURE_ROOT}"
"${PYTHON_BIN}" tools/extract_features.py \
  --dataset-root "${DATASET_ROOT}" --manifest "${PROTOCOL_ROOT}/all_runs/train.jsonl" \
  --checkpoint "${ALLRUN_BACKBONE_CKPT}" --output "${ALLRUN_FEATURE_ROOT}/train_all.pt" \
  --camera-id "${CAMERA_ID}" --batch-size 16 --num-workers "${NUM_WORKERS}" --seed "${SEED}" --amp
"${PYTHON_BIN}" tools/extract_features.py \
  --dataset-root "${DATASET_ROOT}" --manifest "${PROTOCOL_ROOT}/all_runs/test_all.jsonl" \
  --checkpoint "${ALLRUN_BACKBONE_CKPT}" --output "${ALLRUN_FEATURE_ROOT}/test_all.pt" \
  --completion-marker "${ALLRUN_FEATURE_ROOT}/completed.json" \
  --camera-id "${CAMERA_ID}" --batch-size 16 --num-workers "${NUM_WORKERS}" --seed "${SEED}" --amp
