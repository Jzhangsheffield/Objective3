#!/bin/bash
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^(A|D|J|M)$ ]]; then
  echo "Usage: bash slurm/submit_one_fold.sh A|D|J|M" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export TEST_PARTICIPANT="$1"
source "${SCRIPT_DIR}/config_hpc.sh"
prepare_job=$(sbatch --parsable --export=ALL "${SCRIPT_DIR}/01_prepare_protocols.slurm")
backbone_job=$(sbatch --parsable --export=ALL --dependency="afterok:${prepare_job}" "${SCRIPT_DIR}/02_train_backbone_normal_only.slurm")
feature_job=$(sbatch --parsable --export=ALL --dependency="afterok:${backbone_job}" "${SCRIPT_DIR}/03_extract_features.slurm")
m0_job=$(sbatch --parsable --export=ALL --dependency="afterok:${feature_job}" "${SCRIPT_DIR}/04_train_m0.slurm")
context_job=$(sbatch --parsable --export=ALL --dependency="afterok:${m0_job}" "${SCRIPT_DIR}/05_train_context_models.slurm")
summary_job=$(sbatch --parsable --export=ALL --dependency="afterok:${context_job}" "${SCRIPT_DIR}/06_summarize_results.slurm")
echo "${TEST_PARTICIPANT}: prepare=${prepare_job} backbone=${backbone_job} features=${feature_job} m0=${m0_job} context=${context_job} summary=${summary_job}" >&2
echo "${summary_job}"
