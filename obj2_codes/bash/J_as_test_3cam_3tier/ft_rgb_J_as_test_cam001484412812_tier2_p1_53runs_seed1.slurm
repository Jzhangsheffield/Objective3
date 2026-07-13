#!/bin/bash
#SBATCH --job-name=ft_rgb_J_cam001484_tier2
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --time=3-23:00:00
#SBATCH --output=/mnt/parscratch/users/mes19jz/objective3/obj2_codes/ft_and_test/out/slurm-%x-%j.out
#SBATCH --error=/mnt/parscratch/users/mes19jz/objective3/obj2_codes/ft_and_test/out/slurm-%x-%j.err

set -euo pipefail

# ============================================================
# RGB finetuning for J_as_test, camera 001484412812, tier2
#
# Pretraining source:
#   results/rgb_J_as_test_cam001484412812_p1_26runs
#
# This script runs only tier2:
#   - find 26 checkpoint_0200.pth files under the tier2 pretraining directory
#   - run scratch_full baseline when RUN_SCRATCH_BASELINE=1
#   - run pretrained head_only when RUN_HEAD_ONLY=1
#   - run pretrained full fine-tuning when RUN_FULL=1
#
# Tier class counts from Stage_2_Mapstyle_Dataset/label_map.json:
#   tier1=14, tier2=27, tier3=31
# ============================================================

# ------------------------------
# 0) Optional environment setup
# ------------------------------
# Uncomment / edit these lines on your cluster if needed.
# module purge
# module load Anaconda3/2022.05
# module load cuDNN/8.9.2.26-CUDA-12.1.1
# source activate pytorch

ANACONDA_MODULE="${ANACONDA_MODULE:-Anaconda3/2022.05}"
CUDNN_MODULE="${CUDNN_MODULE:-cuDNN/8.9.2.26-CUDA-12.1.1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pytorch}"
SKIP_ENV_SETUP="${SKIP_ENV_SETUP:-0}"

if [[ "${SKIP_ENV_SETUP}" != "1" ]]; then
  if [[ -n "${ANACONDA_MODULE}" ]]; then module load "${ANACONDA_MODULE}"; fi
  if [[ -n "${CUDNN_MODULE}" ]]; then module load "${CUDNN_MODULE}"; fi
  if [[ -n "${CONDA_ENV_NAME}" ]]; then source activate "${CONDA_ENV_NAME}"; fi
fi

# ------------------------------
# 1) Project / data paths
# ------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective3/obj2_codes}"
SCRIPT_PATH="${SCRIPT_PATH:-${PROJECT_ROOT}/ft_and_test/train_mapstyle_finetune_and_test.py}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Stage_2_Mapstyle_Dataset}"
LABEL_MAP_JSON="${LABEL_MAP_JSON:-${DATASET_ROOT}/label_map.json}"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-${DATASET_ROOT}/J_as_test/train_manifest.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-${DATASET_ROOT}/J_as_test/test_manifest.jsonl}"
TEST_MANIFEST="${TEST_MANIFEST:-${DATASET_ROOT}/J_as_test/test_manifest.jsonl}"

CAMERA_ID="${CAMERA_ID:-001484412812}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-${PROJECT_ROOT}/results/rgb_J_as_test_cam001484412812_p1_26runs}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_0200.pth}"
EXPECTED_NUM_CKPTS_PER_TIER="${EXPECTED_NUM_CKPTS_PER_TIER:-26}"
ALLOW_CKPT_COUNT_MISMATCH="${ALLOW_CKPT_COUNT_MISMATCH:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/ft_rgb_J_as_test_cam001484412812_p1_26runs_seed${SEED:-1}}"
SAVE_ROOT="${SAVE_ROOT:-${OUTPUT_ROOT}/weights}"
DATAMAP_ROOT="${DATAMAP_ROOT:-${OUTPUT_ROOT}/datamaps}"
CKPT_LIST_ROOT="${CKPT_LIST_ROOT:-${OUTPUT_ROOT}/checkpoint_lists}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/ft_and_test/out}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"

# ------------------------------
# 2) Run selection
# ------------------------------
TIER_LIST="${TIER_LIST:-tier2}"
RUN_SCRATCH_BASELINE="${RUN_SCRATCH_BASELINE:-1}"
RUN_HEAD_ONLY="${RUN_HEAD_ONLY:-1}"
RUN_FULL="${RUN_FULL:-1}"
# ------------------------------
# 2b) Post-finetuning test selection
# ------------------------------
RUN_TEST_AFTER_FINETUNE="${RUN_TEST_AFTER_FINETUNE:-1}"
TEST_MANIFESTS="${TEST_MANIFESTS:-${TEST_MANIFEST}}"
TEST_CHECKPOINT_NAMES="${TEST_CHECKPOINT_NAMES:-best_val.pth last.pth}"
TEST_FINETUNE_MODES="${TEST_FINETUNE_MODES:-scratch_full head_only full}"
TEST_RESULTS_ROOT="${TEST_RESULTS_ROOT:-${OUTPUT_ROOT}/test_results}"
ALLOW_MISSING_TEST_WEIGHTS="${ALLOW_MISSING_TEST_WEIGHTS:-1}"
NUM_WORKERS_TEST="${NUM_WORKERS_TEST:-8}"
PREFETCH_FACTOR_TEST="${PREFETCH_FACTOR_TEST:-2}"

# ------------------------------
# 3) Dataset / model parameters
# ------------------------------
USE_MODALITY="${USE_MODALITY:-rgb}"
N_FRAMES="${N_FRAMES:-16}"
MODEL_DEPTH="${MODEL_DEPTH:-18}"
L2_NORMALIZE_BEFORE_FC="${L2_NORMALIZE_BEFORE_FC:-0}"
RGB_SIZE="${RGB_SIZE:-224}"
DEPTH_SIZE="${DEPTH_SIZE:-224}"
RGB_MEAN=( 0.5369 0.5295 0.5208 )
RGB_STD=( 0.2311 0.2360 0.2363 )

# ------------------------------
# 4) RGB augmentation parameters
# ------------------------------
RGB_APPLY_SPATIAL_AUG="${RGB_APPLY_SPATIAL_AUG:-1}"
RRC_SCALE_MIN="${RRC_SCALE_MIN:-0.6}"
RRC_SCALE_MAX="${RRC_SCALE_MAX:-1.0}"
RRC_RATIO_MIN="${RRC_RATIO_MIN:-0.75}"
RRC_RATIO_MAX="${RRC_RATIO_MAX:-1.3333333333}"
RGB_HFLIP_P="${RGB_HFLIP_P:-0.0}"
RGB_VFLIP_P="${RGB_VFLIP_P:-0.0}"
RGB_JITTER_P="${RGB_JITTER_P:-0.5}"
RGB_JITTER_BRIGHTNESS="${RGB_JITTER_BRIGHTNESS:-0.24}"
RGB_JITTER_CONTRAST="${RGB_JITTER_CONTRAST:-0.24}"
RGB_JITTER_SATURATION="${RGB_JITTER_SATURATION:-0.24}"
RGB_JITTER_HUE="${RGB_JITTER_HUE:-0.16}"
RGB_GRAY_P="${RGB_GRAY_P:-0.5}"
RGB_BLUR_P="${RGB_BLUR_P:-0.0}"
RGB_BLUR_KERNEL="${RGB_BLUR_KERNEL:-5}"
RGB_BLUR_SIGMA_MIN="${RGB_BLUR_SIGMA_MIN:-0.1}"
RGB_BLUR_SIGMA_MAX="${RGB_BLUR_SIGMA_MAX:-1.0}"
DISABLE_TRAIN_AUGMENTATION="${DISABLE_TRAIN_AUGMENTATION:-0}"

# ------------------------------
# 5) DataLoader / validation parameters
# ------------------------------
NUM_WORKERS_TRAIN="${NUM_WORKERS_TRAIN:-8}"
NUM_WORKERS_VAL="${NUM_WORKERS_VAL:-6}"
PREFETCH_FACTOR_TRAIN="${PREFETCH_FACTOR_TRAIN:-2}"
PREFETCH_FACTOR_VAL="${PREFETCH_FACTOR_VAL:-2}"
DISABLE_VAL="${DISABLE_VAL:-0}"

# ------------------------------
# 6) Optimizer / training control
# ------------------------------
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
OPTIMIZER="${OPTIMIZER:-adamw}"
ADAMW_BETA1="${ADAMW_BETA1:-0.9}"
ADAMW_BETA2="${ADAMW_BETA2:-0.999}"
ADAMW_EPS="${ADAMW_EPS:-1e-8}"
USE_COSINE_LR="${USE_COSINE_LR:-0}"
SCHEDULES=( ${SCHEDULES:-50 75} )
ENABLE_AMP="${ENABLE_AMP:-0}"
SAVE_PERIOD="${SAVE_PERIOD:-20}"
BEST_AFTER_EPOCH="${BEST_AFTER_EPOCH:-0}"

# ------------------------------
# 7) Finetune LR / pretrained loading
# ------------------------------
HEAD_ONLY_HEAD_LR="${HEAD_ONLY_HEAD_LR:-1e-3}"
FULL_BACKBONE_LR="${FULL_BACKBONE_LR:-1e-4}"
FULL_HEAD_LR="${FULL_HEAD_LR:-1e-3}"
USE_DISCRIMINATIVE_LR_FOR_FULL="${USE_DISCRIMINATIVE_LR_FOR_FULL:-1}"
KEEP_PRETRAINED_HEAD="${KEEP_PRETRAINED_HEAD:-0}"
PRETRAINED_STRICT="${PRETRAINED_STRICT:-0}"
PRETRAINED_TAG_MODE="${PRETRAINED_TAG_MODE:-relative_to_anchor}"
PRETRAINED_TAG_LAST_K="${PRETRAINED_TAG_LAST_K:-3}"
PRETRAINED_TAG_ANCHOR="${PRETRAINED_TAG_ANCHOR:-rgb_J_as_test_cam001484412812_p1_26runs}"

# ------------------------------
# 8) Class imbalance options
# ------------------------------
USE_WEIGHTED_SAMPLER="${USE_WEIGHTED_SAMPLER:-0}"
SAMPLER_MODE="${SAMPLER_MODE:-sqrt_inv}"
USE_WEIGHTED_CE="${USE_WEIGHTED_CE:-0}"
WEIGHT_METHOD="${WEIGHT_METHOD:-class_balanced}"
CB_BETA="${CB_BETA:-0.999}"
WEIGHT_NORMALIZE_MEAN="${WEIGHT_NORMALIZE_MEAN:-0}"
USE_FOCAL="${USE_FOCAL:-0}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
FOCAL_USE_ALPHA="${FOCAL_USE_ALPHA:-0}"

mkdir -p "${SAVE_ROOT}" "${DATAMAP_ROOT}" "${CKPT_LIST_ROOT}" "${LOG_ROOT}" "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

append_bool_optional_arg() {
  local value="$1"
  local true_flag="$2"
  local false_flag="$3"
  case "${value}" in
    1|true|TRUE|yes|YES) COMMON_ARGS+=("${true_flag}") ;;
    0|false|FALSE|no|NO) COMMON_ARGS+=("${false_flag}") ;;
    "") ;;
    *) echo "[Error] Invalid boolean value for ${true_flag}/${false_flag}: ${value}" >&2; exit 2 ;;
  esac
}

append_store_true_arg() {
  local value="$1"
  local flag="$2"
  case "${value}" in
    1|true|TRUE|yes|YES) COMMON_ARGS+=("${flag}") ;;
    0|false|FALSE|no|NO|"") ;;
    *) echo "[Error] Invalid boolean value for ${flag}: ${value}" >&2; exit 2 ;;
  esac
}

num_classes_for_tier() {
  case "$1" in
    tier1) echo 14 ;;
    tier2) echo 27 ;;
    tier3) echo 31 ;;
    *) echo "[Error] Unsupported tier: $1" >&2; exit 2 ;;
  esac
}

maybe_full_lr_args() {
  if [[ "${USE_DISCRIMINATIVE_LR_FOR_FULL}" == "1" ]]; then
    FULL_LR_ARGS=(--use_discriminative_lr --backbone_learning_rate "${FULL_BACKBONE_LR}" --head_learning_rate "${FULL_HEAD_LR}")
  else
    FULL_LR_ARGS=(--backbone_learning_rate "${FULL_BACKBONE_LR}" --head_learning_rate "${FULL_HEAD_LR}")
  fi
}

print_command() {
  printf '[Command] '
  printf '%q ' "$@"
  printf '\n'
}

build_common_args() {
  local tier_mode="$1"
  local num_classes="$2"
  local sampler_tier="${SAMPLER_TIER:-${tier_mode}}"

  COMMON_ARGS=(
    --run_mode train
    --dataset_root "${DATASET_ROOT}"
    --label_map_json "${LABEL_MAP_JSON}"
    --train_manifest "${TRAIN_MANIFEST}"
    --val_manifest "${VAL_MANIFEST}"
    --use_modality "${USE_MODALITY}"
    --tier_mode "${tier_mode}"
    --n_frames "${N_FRAMES}"
    --rgb_camera_id "${CAMERA_ID}"
    --num_classes "${num_classes}"
    --model_depth "${MODEL_DEPTH}"
    --rgb_size "${RGB_SIZE}"
    --depth_size "${DEPTH_SIZE}"
    --rgb_mean "${RGB_MEAN[@]}"
    --rgb_std "${RGB_STD[@]}"
    --rrc_scale_min "${RRC_SCALE_MIN}"
    --rrc_scale_max "${RRC_SCALE_MAX}"
    --rrc_ratio_min "${RRC_RATIO_MIN}"
    --rrc_ratio_max "${RRC_RATIO_MAX}"
    --rgb_hflip_p "${RGB_HFLIP_P}"
    --rgb_vflip_p "${RGB_VFLIP_P}"
    --rgb_jitter_p "${RGB_JITTER_P}"
    --rgb_jitter_brightness "${RGB_JITTER_BRIGHTNESS}"
    --rgb_jitter_contrast "${RGB_JITTER_CONTRAST}"
    --rgb_jitter_saturation "${RGB_JITTER_SATURATION}"
    --rgb_jitter_hue "${RGB_JITTER_HUE}"
    --rgb_gray_p "${RGB_GRAY_P}"
    --rgb_blur_p "${RGB_BLUR_P}"
    --rgb_blur_kernel "${RGB_BLUR_KERNEL}"
    --rgb_blur_sigma_min "${RGB_BLUR_SIGMA_MIN}"
    --rgb_blur_sigma_max "${RGB_BLUR_SIGMA_MAX}"
    --num_workers_train "${NUM_WORKERS_TRAIN}"
    --num_workers_val "${NUM_WORKERS_VAL}"
    --prefetch_factor_train "${PREFETCH_FACTOR_TRAIN}"
    --prefetch_factor_val "${PREFETCH_FACTOR_VAL}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --learning_rate "${LEARNING_RATE}"
    --momentum "${MOMENTUM}"
    --weight_decay "${WEIGHT_DECAY}"
    --optimizer "${OPTIMIZER}"
    --adamw_beta1 "${ADAMW_BETA1}"
    --adamw_beta2 "${ADAMW_BETA2}"
    --adamw_eps "${ADAMW_EPS}"
    --schedules "${SCHEDULES[@]}"
    --seed "${SEED}"
    --pretrained_tag_mode "${PRETRAINED_TAG_MODE}"
    --pretrained_tag_last_k "${PRETRAINED_TAG_LAST_K}"
    --pretrained_tag_anchor "${PRETRAINED_TAG_ANCHOR}"
    --sampler_tier "${sampler_tier}"
    --sampler_mode "${SAMPLER_MODE}"
    --weight_method "${WEIGHT_METHOD}"
    --cb_beta "${CB_BETA}"
    --focal_gamma "${FOCAL_GAMMA}"
    --save_period "${SAVE_PERIOD}"
    --best_after_epoch "${BEST_AFTER_EPOCH}"
  )

  append_bool_optional_arg "${RGB_APPLY_SPATIAL_AUG}" "--rgb_apply_spatial_aug" "--no-rgb_apply_spatial_aug"
  append_store_true_arg "${DISABLE_TRAIN_AUGMENTATION}" "--disable_train_augmentation"
  append_store_true_arg "${DISABLE_VAL}" "--disable_val"
  append_store_true_arg "${L2_NORMALIZE_BEFORE_FC}" "--l2_normalize_before_fc"
  append_store_true_arg "${USE_COSINE_LR}" "--cos"
  append_store_true_arg "${ENABLE_AMP}" "--enable_amp"
  append_store_true_arg "${KEEP_PRETRAINED_HEAD}" "--keep_pretrained_head"
  append_store_true_arg "${PRETRAINED_STRICT}" "--pretrained_strict"
  append_store_true_arg "${USE_WEIGHTED_SAMPLER}" "--use_weighted_sampler"
  append_store_true_arg "${USE_WEIGHTED_CE}" "--use_weighted_ce"
  append_store_true_arg "${WEIGHT_NORMALIZE_MEAN}" "--weight_normalize_mean"
  append_store_true_arg "${USE_FOCAL}" "--use_focal"
  append_store_true_arg "${FOCAL_USE_ALPHA}" "--focal_use_alpha"
}

run_train() {
  local label="$1"
  shift
  echo "======================================"
  echo "Start ${label}"
  echo "PROJECT_ROOT    = ${PROJECT_ROOT}"
  echo "SCRIPT_PATH     = ${SCRIPT_PATH}"
  echo "DATASET_ROOT    = ${DATASET_ROOT}"
  echo "CAMERA_ID       = ${CAMERA_ID}"
  echo "TRAIN_MANIFEST  = ${TRAIN_MANIFEST}"
  echo "VAL_MANIFEST    = ${VAL_MANIFEST}"
  echo "PRETRAIN_ROOT   = ${PRETRAIN_ROOT}"
  echo "OUTPUT_ROOT     = ${OUTPUT_ROOT}"
  echo "RGB_MEAN        = ${RGB_MEAN[*]}"
  echo "RGB_STD         = ${RGB_STD[*]}"
  echo "======================================"

  local cmd=("${PYTHON_BIN}" -u "${SCRIPT_PATH}" "$@")
  print_command "${cmd[@]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DryRun] ${label} command constructed; training skipped."
    return 0
  fi
  "${cmd[@]}"
  echo "Finished ${label}"
}


run_test_for_mode_checkpoint() {
  local tier_mode="$1"
  local num_classes="$2"
  local ft_mode="$3"
  local ckpt_name="$4"
  local search_root="${SAVE_ROOT}/${tier_mode}/${ft_mode}"
  local ckpt_tag="${ckpt_name%.pth}"
  local results_dir="${TEST_RESULTS_ROOT}/${tier_mode}/${ft_mode}"
  local results_csv="${results_dir}/${ckpt_tag}_test_results.csv"

  if [[ ! -d "${search_root}" ]]; then
    if [[ "${ALLOW_MISSING_TEST_WEIGHTS}" == "1" ]]; then
      echo "[Warn] Skip test: missing fine-tune directory ${search_root}"
      return 0
    fi
    echo "[Error] Missing fine-tune directory for test: ${search_root}" >&2
    exit 1
  fi

  mapfile -t TEST_WEIGHTS < <(find "${search_root}" -type f -name "${ckpt_name}" | sort)
  if [[ "${#TEST_WEIGHTS[@]}" -eq 0 ]]; then
    if [[ "${ALLOW_MISSING_TEST_WEIGHTS}" == "1" ]]; then
      echo "[Warn] Skip test: no ${ckpt_name} found under ${search_root}"
      return 0
    fi
    echo "[Error] No ${ckpt_name} found under ${search_root}" >&2
    exit 1
  fi

  mkdir -p "${results_dir}"
  read -r -a TEST_MANIFEST_ARRAY <<< "${TEST_MANIFESTS}"

  local cmd=(
    "${PYTHON_BIN}" -u "${SCRIPT_PATH}"
    --run_mode test
    --dataset_root "${DATASET_ROOT}"
    --label_map_json "${LABEL_MAP_JSON}"
    --test_manifest "${TEST_MANIFEST_ARRAY[@]}"
    --test_weight_paths "${TEST_WEIGHTS[@]}"
    --test_results_csv "${results_csv}"
    --save_path "${results_dir}"
    --use_modality "${USE_MODALITY}"
    --tier_mode "${tier_mode}"
    --n_frames "${N_FRAMES}"
    --rgb_camera_id "${CAMERA_ID}"
    --num_classes "${num_classes}"
    --model_depth "${MODEL_DEPTH}"
    --rgb_size "${RGB_SIZE}"
    --depth_size "${DEPTH_SIZE}"
    --rgb_mean "${RGB_MEAN[@]}"
    --rgb_std "${RGB_STD[@]}"
    --num_workers_test "${NUM_WORKERS_TEST}"
    --prefetch_factor_test "${PREFETCH_FACTOR_TEST}"
    --batch_size "${BATCH_SIZE}"
    --seed "${SEED}"
  )

  if [[ "${L2_NORMALIZE_BEFORE_FC}" == "1" ]]; then
    cmd+=(--l2_normalize_before_fc)
  fi
  if [[ "${ENABLE_AMP}" == "1" ]]; then
    cmd+=(--enable_amp)
  fi

  echo "======================================"
  echo "Start test: tier=${tier_mode} mode=${ft_mode} checkpoint=${ckpt_name}"
  echo "Search root      = ${search_root}"
  echo "Num test weights = ${#TEST_WEIGHTS[@]}"
  echo "Test manifests   = ${TEST_MANIFESTS}"
  echo "Results CSV      = ${results_csv}"
  echo "======================================"
  print_command "${cmd[@]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DryRun] test command constructed; testing skipped."
    return 0
  fi

  "${cmd[@]}"
  echo "Finished test: tier=${tier_mode} mode=${ft_mode} checkpoint=${ckpt_name}"
}

run_post_finetune_tests() {
  local tier_mode="$1"
  local num_classes="$2"

  if [[ "${RUN_TEST_AFTER_FINETUNE}" != "1" ]]; then
    echo "Post-finetuning tests disabled: RUN_TEST_AFTER_FINETUNE=${RUN_TEST_AFTER_FINETUNE}"
    return 0
  fi

  for ft_mode in ${TEST_FINETUNE_MODES}; do
    case "${ft_mode}" in
      scratch_full)
        [[ "${RUN_SCRATCH_BASELINE}" == "1" ]] || { echo "[Skip] scratch_full was not trained."; continue; }
        ;;
      head_only)
        [[ "${RUN_HEAD_ONLY}" == "1" ]] || { echo "[Skip] head_only was not trained."; continue; }
        ;;
      full)
        [[ "${RUN_FULL}" == "1" ]] || { echo "[Skip] full was not trained."; continue; }
        ;;
      *)
        echo "[Error] Unsupported TEST_FINETUNE_MODES entry: ${ft_mode}" >&2
        exit 2
        ;;
    esac

    for ckpt_name in ${TEST_CHECKPOINT_NAMES}; do
      run_test_for_mode_checkpoint "${tier_mode}" "${num_classes}" "${ft_mode}" "${ckpt_name}"
    done
  done
}
if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" -c "import sys; print(sys.executable); import utils_; import loss; import backbone; print('imports ok')"
fi

maybe_full_lr_args

for tier_mode in ${TIER_LIST}; do
  num_classes="$(num_classes_for_tier "${tier_mode}")"
  tier_pretrain_parent="${PRETRAIN_ROOT}/${tier_mode}"
  checkpoint_list_txt="${CKPT_LIST_ROOT}/${tier_mode}_${CHECKPOINT_NAME}_paths.txt"

  echo
  echo "============================================================"
  echo "Tier: ${tier_mode} | num_classes=${num_classes}"
  echo "Pretrain parent: ${tier_pretrain_parent}"
  echo "============================================================"

  if [[ ! -d "${tier_pretrain_parent}" ]]; then
    echo "[Error] tier pretrain parent does not exist: ${tier_pretrain_parent}" >&2
    exit 1
  fi

  mapfile -t CKPTS < <(find "${tier_pretrain_parent}" -type f -name "${CHECKPOINT_NAME}" | sort)
  num_ckpts=${#CKPTS[@]}
  printf "%s\n" ${CKPTS[@]+"${CKPTS[@]}"} > "${checkpoint_list_txt}"

  if [[ "${num_ckpts}" -eq 0 ]]; then
    echo "[Error] No ${CHECKPOINT_NAME} files found under: ${tier_pretrain_parent}" >&2
    exit 1
  fi
  if [[ "${num_ckpts}" -ne "${EXPECTED_NUM_CKPTS_PER_TIER}" && "${ALLOW_CKPT_COUNT_MISMATCH}" != "1" ]]; then
    echo "[Error] Expected ${EXPECTED_NUM_CKPTS_PER_TIER} ${CHECKPOINT_NAME} files for ${tier_mode}, but found ${num_ckpts}." >&2
    echo "        Check ${checkpoint_list_txt}, or set ALLOW_CKPT_COUNT_MISMATCH=1 intentionally." >&2
    exit 1
  fi
  for i in "${!CKPTS[@]}"; do
    printf "[%s %02d] %s\n" "${tier_mode}" "$((i + 1))" "${CKPTS[$i]}"
  done

  build_common_args "${tier_mode}" "${num_classes}"

  if [[ "${RUN_SCRATCH_BASELINE}" == "1" ]]; then
    run_train "${tier_mode}_scratch_full" \
      "${COMMON_ARGS[@]}" \
      --save_path "${SAVE_ROOT}/${tier_mode}/scratch_full" \
      --datamap_csv_path "${DATAMAP_ROOT}/${tier_mode}/scratch_full" \
      --finetune_mode full \
      "${FULL_LR_ARGS[@]}"
  fi

  if [[ "${RUN_HEAD_ONLY}" == "1" ]]; then
    run_train "${tier_mode}_pretrained_head_only" \
      "${COMMON_ARGS[@]}" \
      --pretrained_weight_paths "${CKPTS[@]}" \
      --save_path "${SAVE_ROOT}/${tier_mode}/head_only" \
      --datamap_csv_path "${DATAMAP_ROOT}/${tier_mode}/head_only" \
      --finetune_mode head_only \
      --head_learning_rate "${HEAD_ONLY_HEAD_LR}"
  fi

  if [[ "${RUN_FULL}" == "1" ]]; then
    run_train "${tier_mode}_pretrained_full" \
      "${COMMON_ARGS[@]}" \
      --pretrained_weight_paths "${CKPTS[@]}" \
      --save_path "${SAVE_ROOT}/${tier_mode}/full" \
      --datamap_csv_path "${DATAMAP_ROOT}/${tier_mode}/full" \
      --finetune_mode full \
      "${FULL_LR_ARGS[@]}"
  fi
  run_post_finetune_tests "${tier_mode}" "${num_classes}"
done

echo "All finetuning experiments finished."
echo "Output root: ${OUTPUT_ROOT}"

