#!/bin/bash
set -euo pipefail
if [[ $# -ne 2 || ! "$1" =~ ^(A|D|J|M)$ || ! "$2" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash slurm/submit_normal_only_one_fold.sh A|D|J|M SEED" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export TEST_PARTICIPANT="$1"
export SEED="$2"
unset FOLD_ROOT PROTOCOL_ROOT RUN_ROOT BACKBONE_OUTPUT BACKBONE_CKPT
unset FEATURE_ROOT MODEL_ROOT E2E_ROOT E2E_TIER3_OUTPUT
unset E2E_NODE_SCRATCH_OUTPUT E2E_NODE_TRANSFER_OUTPUT NORMAL_FOLD_SUMMARY_ROOT
source "${SCRIPT_DIR}/config_hpc.sh"

protocol_args=(--parsable --export=ALL)
if [[ -n "${UPSTREAM_DEPENDENCY:-}" ]]; then
  protocol_args+=(--dependency="afterok:${UPSTREAM_DEPENDENCY}")
fi
protocol_job=$(sbatch "${protocol_args[@]}" "${SCRIPT_DIR}/13_prepare_protocols_all_runs_safe.slurm")
backbone_job=$(sbatch --parsable --export=ALL --dependency="afterok:${protocol_job}" "${SCRIPT_DIR}/25_train_backbone_normal_only_safe.slurm")
scratch_job=$(sbatch --parsable --export=ALL --dependency="afterok:${protocol_job}" "${SCRIPT_DIR}/09_train_e2e_node_scratch.slurm")
feature_job=$(sbatch --parsable --export=ALL --dependency="afterok:${backbone_job}" "${SCRIPT_DIR}/26_extract_features_normal_only_safe.slurm")
tier3_job=$(sbatch --parsable --export=ALL --dependency="afterok:${backbone_job}" "${SCRIPT_DIR}/08_evaluate_e2e_tier3_existing.slurm")
transfer_job=$(sbatch --parsable --export=ALL --dependency="afterok:${backbone_job}" "${SCRIPT_DIR}/10_train_e2e_node_from_tier3.slurm")
m0_job=$(sbatch --parsable --export=ALL --dependency="afterok:${feature_job}" "${SCRIPT_DIR}/27_train_normal_only_m0.slurm")
context_job=$(sbatch --parsable --export=ALL --dependency="afterok:${m0_job}" "${SCRIPT_DIR}/28_train_normal_only_context_models.slurm")
summary_dependency="${context_job}:${tier3_job}:${scratch_job}:${transfer_job}"
summary_job=$(sbatch --parsable --export=ALL --dependency="afterok:${summary_dependency}" "${SCRIPT_DIR}/29_summarize_normal_only_fold.slurm")
echo "${TEST_PARTICIPANT} seed=${SEED} normal-only: protocol=${protocol_job} backbone=${backbone_job} features=${feature_job} m0=${m0_job} context=${context_job} tier3=${tier3_job} scratch=${scratch_job} transfer=${transfer_job} summary=${summary_job}" >&2
echo "${summary_job}"
