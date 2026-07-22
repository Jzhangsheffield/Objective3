#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
final_jobs=()
for participant in A D M; do
  final_jobs+=("$(bash "${SCRIPT_DIR}/submit_additional_e2e_one_fold.sh" "${participant}")")
done
dependency=$(IFS=:; echo "${final_jobs[*]}")
cross_summary_job=$(sbatch --parsable --export=ALL \
  --dependency="afterok:${dependency}" \
  "${SCRIPT_DIR}/12_summarize_all_models_cross_person.slurm")
echo "A/D/M fold summaries=${dependency} all-model cross-person summary=${cross_summary_job}"

