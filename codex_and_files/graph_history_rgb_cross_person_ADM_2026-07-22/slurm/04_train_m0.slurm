#!/bin/bash
#SBATCH --job-name=gh_m0_LOSO_1484
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=/mnt/parscratch/users/mes19jz/objective3/obj2_codes/ft_and_test/out/slurm-%x-%j.out
#SBATCH --error=/mnt/parscratch/users/mes19jz/objective3/obj2_codes/ft_and_test/out/slurm-%x-%j.err
set -euo pipefail
PACKAGE_ROOT="${PACKAGE_ROOT:-/mnt/parscratch/users/mes19jz/objective3/codex_and_files/graph_history_rgb_cross_person_ADM_2026-07-22}"
source "${PACKAGE_ROOT}/slurm/config_hpc.sh"
setup_hpc_environment
"${PYTHON_BIN}" tools/train_history_model.py \
  --model m0 --train-scope "${TRAIN_SCOPE}" --protocol-root "${PROTOCOL_ROOT}" \
  --train-cache "${FEATURE_ROOT}/train_all.pt" --test-cache "${FEATURE_ROOT}/test_all.pt" \
  --task-graph "${TASK_GRAPH}" --relation-matrix "${RELATION_MATRIX}" \
  --output-root "${MODEL_ROOT}" --epochs "${HISTORY_EPOCHS}" --batch-size 64 \
  --num-workers "${NUM_WORKERS}" --seed "${SEED}"
