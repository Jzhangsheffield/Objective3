#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
jobs=()
for participant in A D J M; do
  for seed in 1 2 42; do
    pair=$(bash "${SCRIPT_DIR}/submit_dynamic_epoch_shuffle_one_fold.sh" \
      "${participant}" "${seed}" both)
    IFS=: read -r normal_job allrun_job <<< "${pair}"
    jobs+=("${normal_job}" "${allrun_job}")
  done
done
dependency=$(IFS=:; echo "${jobs[*]}")
summary_job=$(sbatch --parsable --export=ALL --dependency="afterok:${dependency}" \
  "${SCRIPT_DIR}/38_summarize_dynamic_epoch_shuffle_ADJM_3seeds.slurm")
echo "Dynamic epoch-shuffle training jobs=${dependency}" >&2
echo "Dynamic epoch-shuffle ADJM summary=${summary_job}" >&2
echo "${summary_job}"
