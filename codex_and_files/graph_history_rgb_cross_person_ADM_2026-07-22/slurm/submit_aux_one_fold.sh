#!/bin/bash
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^(A|D|J|M)$ ]]; then
  echo "Usage: bash slurm/submit_aux_one_fold.sh A|D|J|M" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export TEST_PARTICIPANT="$1"
export TRAIN_SCOPE=all_runs
source "${SCRIPT_DIR}/config_hpc.sh"
if [[ ! -f "${FEATURE_ROOT}/train_all.pt" || ! -f "${FEATURE_ROOT}/test_all.pt" ]]; then
  echo "Feature caches are missing. Complete the main fold pipeline first." >&2
  exit 1
fi
m0_job=$(sbatch --parsable --export=ALL "${SCRIPT_DIR}/04_train_m0.slurm")
context_job=$(sbatch --parsable --export=ALL --dependency="afterok:${m0_job}" "${SCRIPT_DIR}/05_train_context_models.slurm")
summary_job=$(sbatch --parsable --export=ALL --dependency="afterok:${context_job}" "${SCRIPT_DIR}/06_summarize_results.slurm")
echo "${TEST_PARTICIPANT} all-run auxiliary: m0=${m0_job} context=${context_job} summary=${summary_job}"
