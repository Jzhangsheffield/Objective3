#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
final_jobs=()
for participant in A D M; do
  final_jobs+=("$(bash "${SCRIPT_DIR}/submit_all_runs_one_fold.sh" "${participant}")")
done
dependency=$(IFS=:; echo "${final_jobs[*]}")
allrun_job=$(sbatch --parsable --export=ALL --dependency="afterok:${dependency}" "${SCRIPT_DIR}/23_summarize_all_runs_cross_person.slurm")
comparison_job=$(sbatch --parsable --export=ALL --dependency="afterok:${dependency}" "${SCRIPT_DIR}/24_summarize_training_scope_comparison_cross_person.slurm")
echo "A/D/M fold comparisons=${dependency} all-runs summary=${allrun_job} training-scope comparison=${comparison_job}"
