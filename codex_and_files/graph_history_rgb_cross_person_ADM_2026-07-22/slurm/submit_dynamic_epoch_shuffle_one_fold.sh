#!/bin/bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 TEST_PARTICIPANT SEED [normal_only|all_runs|both]" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
participant="$1"
seed="$2"
scope="${3:-both}"
outputs_root="${OUTPUTS_ROOT:-${PACKAGE_ROOT}/outputs}"
camera_id="${CAMERA_ID:-001484412812}"
fold_root="${outputs_root}/${participant}_as_test/cam_${camera_id}"
protocol_root="${fold_root}/protocols"
run_root="${fold_root}/seed_${seed}"
feature_root="${run_root}/features/retrained_normal_only"
model_root="${run_root}/history_models/retrained_normal_only"
allrun_feature_root="${run_root}/features/retrained_all_runs"
allrun_model_root="${run_root}/history_models/retrained_all_runs"
dynamic_model_root="${run_root}/history_models/dynamic_epoch_shuffle"
job_exports="ALL,TEST_PARTICIPANT=${participant},SEED=${seed},OUTPUTS_ROOT=${outputs_root},CAMERA_ID=${camera_id},FOLD_ROOT=${fold_root},PROTOCOL_ROOT=${protocol_root},RUN_ROOT=${run_root},FEATURE_ROOT=${feature_root},MODEL_ROOT=${model_root},ALLRUN_FEATURE_ROOT=${allrun_feature_root},ALLRUN_MODEL_ROOT=${allrun_model_root},DYNAMIC_MODEL_ROOT=${dynamic_model_root}"
case "${scope}" in
  normal_only)
    job=$(sbatch --parsable --export="${job_exports}" \
      "${SCRIPT_DIR}/36_train_dynamic_epoch_shuffle_normal_only.slurm")
    ;;
  all_runs)
    job=$(sbatch --parsable --export="${job_exports}" \
      "${SCRIPT_DIR}/37_train_dynamic_epoch_shuffle_all_runs.slurm")
    ;;
  both)
    normal_job=$(sbatch --parsable --export="${job_exports}" \
      "${SCRIPT_DIR}/36_train_dynamic_epoch_shuffle_normal_only.slurm")
    allrun_job=$(sbatch --parsable --export="${job_exports}" \
      "${SCRIPT_DIR}/37_train_dynamic_epoch_shuffle_all_runs.slurm")
    job="${normal_job}:${allrun_job}"
    ;;
  *)
    echo "Unsupported scope: ${scope}" >&2
    exit 2
    ;;
esac
echo "${job}"
