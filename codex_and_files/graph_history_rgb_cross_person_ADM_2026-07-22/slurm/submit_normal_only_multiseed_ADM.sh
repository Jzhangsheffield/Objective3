#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
final_jobs=()
for participant in A D M; do
  for seed in 2 42; do
    final_jobs+=("$(bash "${SCRIPT_DIR}/submit_normal_only_one_fold.sh" "${participant}" "${seed}")")
  done
done
dependency=$(IFS=:; echo "${final_jobs[*]}")
echo "A/D/M missing normal-only seeds submitted. Final fold summaries=${dependency}" >&2
echo "${dependency}"
