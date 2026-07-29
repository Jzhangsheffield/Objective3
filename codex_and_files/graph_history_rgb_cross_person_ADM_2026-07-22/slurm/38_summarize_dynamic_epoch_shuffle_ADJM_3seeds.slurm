#!/bin/bash
#SBATCH --job-name=gh_dyn_sum
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/parscratch/users/mes19jz/objective3/obj2_codes/ft_and_test/out/slurm-%x-%j.out
#SBATCH --error=/mnt/parscratch/users/mes19jz/objective3/obj2_codes/ft_and_test/out/slurm-%x-%j.err
set -euo pipefail
PACKAGE_ROOT="${PACKAGE_ROOT:-/mnt/parscratch/users/mes19jz/objective3/codex_and_files/graph_history_rgb_cross_person_ADM_2026-07-22}"
source "${PACKAGE_ROOT}/slurm/config_hpc.sh"
setup_hpc_environment
if [[ -f "${DYNAMIC_FOURFOLD_SUMMARY_ROOT}/completed.json" ]]; then
  echo "Completed dynamic epoch-shuffle summary exists; skip without overwrite."
  exit 0
fi
"${PYTHON_BIN}" tools/summarize_dynamic_epoch_shuffle.py \
  --outputs-root "${OUTPUTS_ROOT}" \
  --output-dir "${DYNAMIC_FOURFOLD_SUMMARY_ROOT}" \
  --camera-id "${CAMERA_ID}" \
  --participants A D J M \
  --seeds 1 2 42 \
  --train-scopes normal_only all_runs \
  --require-complete-grid
