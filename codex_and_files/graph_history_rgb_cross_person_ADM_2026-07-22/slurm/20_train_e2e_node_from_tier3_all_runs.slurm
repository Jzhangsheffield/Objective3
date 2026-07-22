#!/bin/bash
#SBATCH --job-name=gh_ar_e2e_nt
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=3-23:00:00
#SBATCH --output=/mnt/parscratch/users/mes19jz/objective3/obj2_codes/ft_and_test/out/slurm-%x-%j.out
#SBATCH --error=/mnt/parscratch/users/mes19jz/objective3/obj2_codes/ft_and_test/out/slurm-%x-%j.err
set -euo pipefail
PACKAGE_ROOT="${PACKAGE_ROOT:-/mnt/parscratch/users/mes19jz/objective3/codex_and_files/graph_history_rgb_cross_person_ADM_2026-07-22}"
source "${PACKAGE_ROOT}/slurm/config_hpc.sh"
setup_hpc_environment
if [[ -f "${ALLRUN_E2E_NODE_TRANSFER_OUTPUT}/completed.json" ]]; then echo "Completed; skip."; exit 0; fi
"${PYTHON_BIN}" tools/train_e2e_node.py \
  --dataset-root "${DATASET_ROOT}" --protocol-root "${PROTOCOL_ROOT}" \
  --task-graph "${TASK_GRAPH}" --relation-matrix "${RELATION_MATRIX}" \
  --output-dir "${ALLRUN_E2E_NODE_TRANSFER_OUTPUT}" --init tier3 \
  --init-checkpoint "${ALLRUN_BACKBONE_CKPT}" --train-scope all_runs \
  --camera-id "${CAMERA_ID}" --epochs "${E2E_NODE_EPOCHS}" --batch-size 16 \
  --num-workers "${NUM_WORKERS}" --learning-rate "${E2E_NODE_LR}" --seed "${SEED}" --amp
