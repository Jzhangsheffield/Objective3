#!/bin/bash
#SBATCH --job-name=gh_ar_summary
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
  --outputs-root "${OUTPUTS_ROOT}" --output-dir "${ALLRUN_FOLD_SUMMARY_ROOT}" \
  --participants "${TEST_PARTICIPANT}" --seeds "${SEED}" \
  --train-scopes all_runs --representation-scopes all_runs --matched-scope-only
