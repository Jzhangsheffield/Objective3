#!/bin/bash
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^(A|D|J|M)$ ]]; then
  echo "Usage: bash slurm/submit_all_runs_one_fold.sh A|D|J|M" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export TEST_PARTICIPANT="$1"
source "${SCRIPT_DIR}/config_hpc.sh"

protocol_job=$(sbatch --parsable --export=ALL "${SCRIPT_DIR}/13_prepare_protocols_all_runs_safe.slurm")
backbone_job=$(sbatch --parsable --export=ALL --dependency="afterok:${protocol_job}" "${SCRIPT_DIR}/14_train_backbone_all_runs.slurm")
scratch_job=$(sbatch --parsable --export=ALL --dependency="afterok:${protocol_job}" "${SCRIPT_DIR}/19_train_e2e_node_scratch_all_runs.slurm")
feature_job=$(sbatch --parsable --export=ALL --dependency="afterok:${backbone_job}" "${SCRIPT_DIR}/15_extract_features_all_runs.slurm")
tier3_job=$(sbatch --parsable --export=ALL --dependency="afterok:${backbone_job}" "${SCRIPT_DIR}/18_evaluate_e2e_tier3_all_runs.slurm")
transfer_job=$(sbatch --parsable --export=ALL --dependency="afterok:${backbone_job}" "${SCRIPT_DIR}/20_train_e2e_node_from_tier3_all_runs.slurm")
m0_job=$(sbatch --parsable --export=ALL --dependency="afterok:${feature_job}" "${SCRIPT_DIR}/16_train_all_runs_m0.slurm")
context_job=$(sbatch --parsable --export=ALL --dependency="afterok:${m0_job}" "${SCRIPT_DIR}/17_train_all_runs_context_models.slurm")
summary_dependency="${context_job}:${tier3_job}:${scratch_job}:${transfer_job}"
summary_job=$(sbatch --parsable --export=ALL --dependency="afterok:${summary_dependency}" "${SCRIPT_DIR}/21_summarize_all_runs_fold.slurm")
comparison_job=$(sbatch --parsable --export=ALL --dependency="afterok:${summary_job}" "${SCRIPT_DIR}/22_summarize_training_scope_comparison_fold.slurm")
echo "${TEST_PARTICIPANT}: protocol=${protocol_job} backbone=${backbone_job} features=${feature_job} m0=${m0_job} context=${context_job} tier3=${tier3_job} scratch=${scratch_job} transfer=${transfer_job} summary=${summary_job} comparison=${comparison_job}" >&2
echo "${comparison_job}"
