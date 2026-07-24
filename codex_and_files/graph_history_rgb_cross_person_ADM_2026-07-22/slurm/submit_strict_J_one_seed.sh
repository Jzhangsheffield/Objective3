#!/bin/bash
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash slurm/submit_strict_J_one_seed.sh SEED" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
seed="$1"
normal_job=$(bash "${SCRIPT_DIR}/submit_normal_only_one_fold.sh" J "${seed}")
comparison_job=$(UPSTREAM_DEPENDENCY="${normal_job}" SEED="${seed}" bash "${SCRIPT_DIR}/submit_all_runs_one_fold.sh" J)
echo "J seed=${seed}: normal-only summary=${normal_job} all-runs/fold-comparison=${comparison_job}" >&2
echo "${comparison_job}"
