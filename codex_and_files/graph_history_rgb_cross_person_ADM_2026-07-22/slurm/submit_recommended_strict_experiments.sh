#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${PACKAGE_ROOT}/outputs}"
CAMERA_ID="${CAMERA_ID:-001484412812}"

for participant in A D M; do
  for seed in 1 2 42; do
    marker="${OUTPUTS_ROOT}/${participant}_as_test/cam_${CAMERA_ID}/seed_${seed}/backbone/all_runs/completed.json"
    if [[ ! -f "${marker}" ]]; then
      echo "Missing existing A/D/M all-runs completion marker: ${marker}" >&2
      exit 1
    fi
  done
done

final_jobs=()
for participant in A D M; do
  for seed in 2 42; do
    final_jobs+=("$(bash "${SCRIPT_DIR}/submit_normal_only_one_fold.sh" "${participant}" "${seed}")")
  done
done
for seed in 1 2 42; do
  final_jobs+=("$(bash "${SCRIPT_DIR}/submit_strict_J_one_seed.sh" "${seed}")")
done
dependency=$(IFS=:; echo "${final_jobs[*]}")
normal_summary=$(sbatch --parsable --export=ALL --dependency="afterok:${dependency}" "${SCRIPT_DIR}/30_summarize_normal_only_ADJM_3seeds.slurm")
allrun_summary=$(sbatch --parsable --export=ALL --dependency="afterok:${dependency}" "${SCRIPT_DIR}/31_summarize_all_runs_ADJM_3seeds.slurm")
comparison_summary=$(sbatch --parsable --export=ALL --dependency="afterok:${dependency}" "${SCRIPT_DIR}/32_summarize_training_scope_comparison_ADJM_3seeds.slurm")
echo "Final fold jobs=${dependency}" >&2
echo "ADJM normal-only summary=${normal_summary} all-runs summary=${allrun_summary} paired comparison=${comparison_summary}" >&2
echo "${comparison_summary}"
