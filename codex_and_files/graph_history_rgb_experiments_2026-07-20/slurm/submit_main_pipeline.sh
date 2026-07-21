#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
prepare_job=$(sbatch --parsable "${SCRIPT_DIR}/01_prepare_protocols.slurm")
feature_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" "${SCRIPT_DIR}/03_extract_features.slurm")
m0_job=$(sbatch --parsable --dependency="afterok:${feature_job}" "${SCRIPT_DIR}/04_train_m0.slurm")
context_job=$(sbatch --parsable --dependency="afterok:${m0_job}" "${SCRIPT_DIR}/05_train_context_models.slurm")
summary_job=$(sbatch --parsable --dependency="afterok:${context_job}" "${SCRIPT_DIR}/06_summarize_results.slurm")
echo "prepare=${prepare_job} features=${feature_job} m0=${m0_job} context=${context_job} summary=${summary_job}"
