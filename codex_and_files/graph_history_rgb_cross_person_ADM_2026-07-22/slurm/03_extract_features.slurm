#!/bin/bash
#SBATCH --job-name=gh_features_LOSO_1484
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=/mnt/parscratch/users/mes19jz/objective3/obj2_codes/ft_and_test/out/slurm-%x-%j.out
#SBATCH --error=/mnt/parscratch/users/mes19jz/objective3/obj2_codes/ft_and_test/out/slurm-%x-%j.err
set -euo pipefail
PACKAGE_ROOT="${PACKAGE_ROOT:-/mnt/parscratch/users/mes19jz/objective3/codex_and_files/graph_history_rgb_cross_person_ADM_2026-07-22}"
source "${PACKAGE_ROOT}/slurm/config_hpc.sh"
setup_hpc_environment
mkdir -p "${FEATURE_ROOT}"
if [[ ! -f "${BACKBONE_CKPT}" ]]; then
  echo "Missing retrained backbone: ${BACKBONE_CKPT}" >&2
  exit 1
fi
"${PYTHON_BIN}" tools/extract_features.py \
  --dataset-root "${DATASET_ROOT}" --manifest "${PROTOCOL_ROOT}/all_runs/train.jsonl" \
  --checkpoint "${BACKBONE_CKPT}" --output "${FEATURE_ROOT}/train_all.pt" \
  --camera-id "${CAMERA_ID}" --batch-size 16 --num-workers "${NUM_WORKERS}" --seed "${SEED}" --amp
"${PYTHON_BIN}" tools/extract_features.py \
  --dataset-root "${DATASET_ROOT}" --manifest "${PROTOCOL_ROOT}/normal_only/test_all.jsonl" \
  --checkpoint "${BACKBONE_CKPT}" --output "${FEATURE_ROOT}/test_all.pt" \
  --camera-id "${CAMERA_ID}" --batch-size 16 --num-workers "${NUM_WORKERS}" --seed "${SEED}" --amp
