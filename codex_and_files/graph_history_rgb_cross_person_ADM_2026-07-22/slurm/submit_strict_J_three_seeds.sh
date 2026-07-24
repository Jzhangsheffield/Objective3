#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
final_jobs=()
for seed in 1 2 42; do
  final_jobs+=("$(bash "${SCRIPT_DIR}/submit_strict_J_one_seed.sh" "${seed}")")
done
dependency=$(IFS=:; echo "${final_jobs[*]}")
echo "J strict normal-only + all-runs seeds 1/2/42 submitted. Final comparisons=${dependency}" >&2
echo "${dependency}"
