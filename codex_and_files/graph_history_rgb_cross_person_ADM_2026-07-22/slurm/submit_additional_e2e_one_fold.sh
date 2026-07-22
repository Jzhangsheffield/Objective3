#!/bin/bash
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^(A|D|J|M)$ ]]; then
  echo "Usage: bash slurm/submit_additional_e2e_one_fold.sh A|D|J|M" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export TEST_PARTICIPANT="$1"
source "${SCRIPT_DIR}/config_hpc.sh"
if [[ ! -f "${BACKBONE_CKPT}" ]]; then
  echo "Existing Tier-3 last.pth is missing: ${BACKBONE_CKPT}" >&2
  exit 1
fi
for required in \
  "${PROTOCOL_ROOT}/normal_only/train.jsonl" \
  "${PROTOCOL_ROOT}/normal_only/test_normal.jsonl" \
  "${PROTOCOL_ROOT}/normal_only/test_fault.jsonl" \
  "${PROTOCOL_ROOT}/normal_only/test_all.jsonl"; do
  [[ -f "${required}" ]] || { echo "Missing existing protocol: ${required}" >&2; exit 1; }
done
for model in m0 m1 m2 m3 m4 m5 m6; do
  checkpoint="${MODEL_ROOT}/normal_only/${model}/last.pth"
  [[ -f "${checkpoint}" ]] || { echo "Missing existing ${model}: ${checkpoint}" >&2; exit 1; }
done
tier3_job=$(sbatch --parsable --export=ALL "${SCRIPT_DIR}/08_evaluate_e2e_tier3_existing.slurm")
scratch_job=$(sbatch --parsable --export=ALL "${SCRIPT_DIR}/09_train_e2e_node_scratch.slurm")
transfer_job=$(sbatch --parsable --export=ALL "${SCRIPT_DIR}/10_train_e2e_node_from_tier3.slurm")
summary_job=$(sbatch --parsable --export=ALL \
  --dependency="afterok:${tier3_job}:${scratch_job}:${transfer_job}" \
  "${SCRIPT_DIR}/11_summarize_all_models_fold.slurm")
echo "${TEST_PARTICIPANT}: tier3_eval=${tier3_job} node_scratch=${scratch_job} node_from_tier3=${transfer_job} unified_summary=${summary_job}" >&2
echo "${summary_job}"

