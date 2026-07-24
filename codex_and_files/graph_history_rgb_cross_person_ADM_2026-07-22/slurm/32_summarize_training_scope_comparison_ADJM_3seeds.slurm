#!/bin/bash
#SBATCH --job-name=gh_cmp_ADJM3
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
"${PYTHON_BIN}" tools/summarize_all_models.py \
  --outputs-root "${OUTPUTS_ROOT}" --output-dir "${FOURFOLD_COMPARISON_ROOT}" \
  --participants A D J M --seeds 1 2 42 \
  --train-scopes normal_only all_runs \
  --representation-scopes normal_only all_runs \
  --matched-scope-only --require-complete-grid
