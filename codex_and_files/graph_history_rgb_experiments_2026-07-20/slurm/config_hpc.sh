#!/bin/bash
# Central HPC configuration. Every value can be overridden with sbatch --export or env.
PACKAGE_ROOT="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Stage_2_Mapstyle_Dataset}"
CAMERA_ID="${CAMERA_ID:-001484412812}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PACKAGE_ROOT}/outputs/J_as_test/cam_${CAMERA_ID}}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${OUTPUT_ROOT}/protocols}"
FEATURE_ROOT="${FEATURE_ROOT:-${OUTPUT_ROOT}/features/existing_last}"
MODEL_ROOT="${MODEL_ROOT:-${OUTPUT_ROOT}/history_models/existing_last}"
TASK_GRAPH="${TASK_GRAPH:-${PACKAGE_ROOT}/assets/integrated_task_graph_latest.json}"
RELATION_MATRIX="${RELATION_MATRIX:-${PACKAGE_ROOT}/assets/integrated_feature_history_matrix.json}"
EXISTING_BACKBONE="${EXISTING_BACKBONE:-/mnt/parscratch/users/mes19jz/objective3/obj2_codes/results/ft_rgb_J_as_test_cam001484412812_p1_26runs_seed1/scratch_full/run_01_scratch/last.pth}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TRAIN_SCOPE="${TRAIN_SCOPE:-normal_only}"
export PACKAGE_ROOT DATASET_ROOT CAMERA_ID OUTPUT_ROOT PROTOCOL_ROOT FEATURE_ROOT MODEL_ROOT
export TASK_GRAPH RELATION_MATRIX EXISTING_BACKBONE PYTHON_BIN SEED NUM_WORKERS TRAIN_SCOPE
export PYTHONPATH="${PACKAGE_ROOT}:${PYTHONPATH:-}"

setup_hpc_environment() {
  if [[ "${SKIP_ENV_SETUP:-0}" != "1" ]]; then
    module load "${ANACONDA_MODULE:-Anaconda3/2022.05}"
    module load "${CUDNN_MODULE:-cuDNN/8.9.2.26-CUDA-12.1.1}"
    set +u
    source activate "${CONDA_ENV_NAME:-pytorch}"
    set -u
  fi
  cd "${PACKAGE_ROOT}"
}
