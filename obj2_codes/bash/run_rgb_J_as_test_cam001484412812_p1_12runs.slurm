#!/bin/bash
#SBATCH --job-name=rgb_J_cam001484_p1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# ============================================================
# RGB SupLoss ablation for J_as_test, camera 001484412812
#
# Runs:
#   3 tiers x 4 ablation configs = 12 training runs
#
# Ablations:
#   1) suploss only              -> contrastive_only
#   2) suploss + rel             -> contrastive_rel
#   3) suploss + proto           -> contrastive_proto
#   4) suploss + rel + proto     -> contrastive_proto_rel
#
# Note:
#   The current dataloader falls back to record["001484412812_rgb"]
#   when record["rgb"] is absent, so no training-code change is needed
#   for this camera.
# ============================================================

# ------------------------------
# 0) Optional environment setup
# ------------------------------
# Uncomment / edit these lines on your cluster if needed.
# module purge
# module load cuda/12.6
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate Pytorch

# ------------------------------
# 1) Project / data paths
# ------------------------------
# Override any variable at submit time if your SLURM node uses Linux mount paths, e.g.:
#   sbatch --export=ALL,PROJECT_ROOT=/path/to/obj2_codes,DATASET_ROOT=/path/to/Stage_2_Mapstyle_Dataset run_rgb_J_as_test_cam001484412812_p1_12runs.slurm
PROJECT_ROOT="${PROJECT_ROOT:-D:/Junxi_data/Objective3_thermal_crimp/obj2_codes}"
PY_SCRIPT="${PY_SCRIPT:-${PROJECT_ROOT}/train/MoCo_main_supcon_mapstyle_varproto_debug_topk_adamw.py}"
DATASET_ROOT="${DATASET_ROOT:-C:/Junxi_data_for_training_speedup/Stage_2_Mapstyle_Dataset}"
TRAIN_MANIFEST_NAME="${TRAIN_MANIFEST_NAME:-J_as_test/train_manifest.jsonl}"
LABEL_MAP_JSON="${LABEL_MAP_JSON:-${DATASET_ROOT}/label_map.json}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/results/rgb_J_as_test_cam001484412812_p1_12runs}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ------------------------------
# 2) SLURM / launch parameters
# ------------------------------
USE_DDP="${USE_DDP:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"
NO_DDP_ARG="--no_ddp"
if [[ "${USE_DDP}" == "1" ]]; then
  NO_DDP_ARG=""
fi

# ------------------------------
# 3) Dataset / label / clip parameters
# ------------------------------
CAMERA_ID="${CAMERA_ID:-001484412812}"
TIER_LIST="${TIER_LIST:-tier1 tier2 tier3}"
N_FRAMES="${N_FRAMES:-16}"

# J_as_test/train_rgb_camera_stats.json, camera 001484412812, scaled to [0,1].
RGB_MEAN="${RGB_MEAN:-0.5369334156645484 0.5294515169750539 0.5207687980507235}"
RGB_STD="${RGB_STD:-0.23107112181743403 0.23603902073259872 0.23630518070624965}"

# ------------------------------
# 4) RGB augmentation parameters
# ------------------------------
RGB_OUT_HW="${RGB_OUT_HW:-224 224}"
RRC_SCALE="${RRC_SCALE:-0.6 1.0}"
RRC_RATIO="${RRC_RATIO:-0.75 1.3333333333}"
RGB_APPLY_SPATIAL_AUG_ARG="${RGB_APPLY_SPATIAL_AUG_ARG:---rgb_apply_spatial_aug}"
RGB_HFLIP_P="${RGB_HFLIP_P:-0.5}"
RGB_VFLIP_P="${RGB_VFLIP_P:-0.5}"
RGB_JITTER_P="${RGB_JITTER_P:-0.5}"
RGB_JITTER_BRIGHTNESS="${RGB_JITTER_BRIGHTNESS:-0.24}"
RGB_JITTER_CONTRAST="${RGB_JITTER_CONTRAST:-0.24}"
RGB_JITTER_SATURATION="${RGB_JITTER_SATURATION:-0.24}"
RGB_JITTER_HUE="${RGB_JITTER_HUE:-0.16}"
RGB_GRAY_P="${RGB_GRAY_P:-0.2}"
RGB_BLUR_P="${RGB_BLUR_P:-0.5}"
RGB_BLUR_KERNEL="${RGB_BLUR_KERNEL:-5}"
RGB_BLUR_SIGMA="${RGB_BLUR_SIGMA:-0.1 1.0}"

# ------------------------------
# 5) DataLoader parameters
# ------------------------------
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-}"
PIN_MEMORY_ARG="${PIN_MEMORY_ARG:-}"
VERIFY_PATHS_ON_INIT_ARG="${VERIFY_PATHS_ON_INIT_ARG:---verify_paths_on_init}"

PROTO_REFRESH_BATCH_SIZE="${PROTO_REFRESH_BATCH_SIZE:-16}"
PROTO_REFRESH_NUM_WORKERS="${PROTO_REFRESH_NUM_WORKERS:-8}"
PROTO_REFRESH_PREFETCH_FACTOR="${PROTO_REFRESH_PREFETCH_FACTOR:-}"
PROTO_REFRESH_PIN_MEMORY_ARG="${PROTO_REFRESH_PIN_MEMORY_ARG:-}"
PROTO_REFRESH_VERIFY_PATHS_ON_INIT_ARG="${PROTO_REFRESH_VERIFY_PATHS_ON_INIT_ARG:---proto_refresh_verify_paths_on_init}"

# ------------------------------
# 6) Model / contrastive parameters
# ------------------------------
MODEL_DEPTH="${MODEL_DEPTH:-18}"
PROJ_DIM="${PROJ_DIM:-128}"
K_QUEUE="${K_QUEUE:-3200}"
TEMPERATURE="${TEMPERATURE:-0.07}"
USE_MLP_ARG="${USE_MLP_ARG:---mlp}"
CONTRASTIVE_LOSS="${CONTRASTIVE_LOSS:-suploss}"
NUM_POSITIVE="${NUM_POSITIVE:-6}"
EXCLUDE_INVALID_QUEUE_ARG="${EXCLUDE_INVALID_QUEUE_ARG:-}"

# ------------------------------
# 7) Ablation / prototype parameters
# ------------------------------
ABLATION_MODES="${ABLATION_MODES:-contrastive_only contrastive_rel contrastive_proto contrastive_proto_rel}"
DEFAULT_NUM_PROTOTYPES="${DEFAULT_NUM_PROTOTYPES:-1}"
NUM_PROTOTYPES_PER_CLASS="${NUM_PROTOTYPES_PER_CLASS:-}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-50}"
RECLUSTER_INTERVAL="${RECLUSTER_INTERVAL:-10}"
LAMBDA_PROTO="${LAMBDA_PROTO:-1.0}"
PROTO_TEMPERATURE="${PROTO_TEMPERATURE:-0.07}"
ENABLE_PROTOTYPE_TEMPERATURE_SCALING_ARG="${ENABLE_PROTOTYPE_TEMPERATURE_SCALING_ARG:-}"
PROTO_TEMPERATURE_EPS="${PROTO_TEMPERATURE_EPS:-1e-6}"
PROTO_KMEANS_RANDOM_STATE="${PROTO_KMEANS_RANDOM_STATE:-42}"
PROTO_KMEANS_N_INIT="${PROTO_KMEANS_N_INIT:-10}"
PROTO_KMEANS_MAX_ITER="${PROTO_KMEANS_MAX_ITER:-300}"

# ------------------------------
# 8) Relative/prototype directional loss parameters
# ------------------------------
LAMBDA_REL="${LAMBDA_REL:-1.0}"
PROTO_EMA_MOMENTUM="${PROTO_EMA_MOMENTUM:-0.99}"
PREVIEW_EMA_MOMENTUM="${PREVIEW_EMA_MOMENTUM:-0.5}"
REL_SAME_MARGIN="${REL_SAME_MARGIN:-0.01}"
REL_DIFF_MARGIN="${REL_DIFF_MARGIN:-0.01}"
REL_SAME_WEIGHT="${REL_SAME_WEIGHT:-1.0}"
REL_DIFF_WEIGHT="${REL_DIFF_WEIGHT:-1.0}"
REL_TOPK_DIFF_CLASSES="${REL_TOPK_DIFF_CLASSES:-3}"

# ------------------------------
# 9) Loss-stage schedule parameters
# ------------------------------
ENABLE_LOSS_STAGE_SCHEDULE_ARG="${ENABLE_LOSS_STAGE_SCHEDULE_ARG:---enable_loss_stage_schedule}"
PROTO_LOSS_START_EPOCH="${PROTO_LOSS_START_EPOCH:-50}"
REL_LOSS_START_EPOCH="${REL_LOSS_START_EPOCH:-50}"
REL_LOSS_END_EPOCH="${REL_LOSS_END_EPOCH:-200}"
REL_LAMBDA_SCHEDULE="${REL_LAMBDA_SCHEDULE:-cosine}"

# ------------------------------
# 10) Optimizer / training control
# ------------------------------
EPOCHS="${EPOCHS:-200}"
START_EPOCH="${START_EPOCH:-0}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
USE_COSINE_LR_ARG="${USE_COSINE_LR_ARG:-}"
SCHEDULE="${SCHEDULE:-50 100 150}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
MOMENTUM="${MOMENTUM:-0.9}"
OPTIMIZER="${OPTIMIZER:-adamw}"
ADAMW_BETAS="${ADAMW_BETAS:-0.9 0.999}"
ADAMW_EPS="${ADAMW_EPS:-1e-8}"
ADAMW_AMSGRAD_ARG="${ADAMW_AMSGRAD_ARG:---no-adamw_amsgrad}"
SEED_ARG="${SEED_ARG:-}"
PRINT_FREQ="${PRINT_FREQ:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
USE_SYNCBN_ARG="${USE_SYNCBN_ARG:---use_syncbn}"
FIND_UNUSED_PARAMETERS_ARG="${FIND_UNUSED_PARAMETERS_ARG:---no-find_unused_parameters}"

# ------------------------------
# 11) Debug parameters
# ------------------------------
DEBUG_MODE_ARG="${DEBUG_MODE_ARG:-}"
DEBUG_LOG_INTERVAL="${DEBUG_LOG_INTERVAL:-20}"
DEBUG_GRAD_STATS_ARG="${DEBUG_GRAD_STATS_ARG:---debug_grad_stats}"
DEBUG_PARAM_UPDATE_STATS_ARG="${DEBUG_PARAM_UPDATE_STATS_ARG:---debug_param_update_stats}"
DEBUG_BATCH_LABEL_STATS_ARG="${DEBUG_BATCH_LABEL_STATS_ARG:---debug_batch_label_stats}"
DEBUG_PROTO_STATS_ARG="${DEBUG_PROTO_STATS_ARG:---debug_proto_stats}"
DEBUG_FEATURE_STATS_ARG="${DEBUG_FEATURE_STATS_ARG:---debug_feature_stats}"
DEBUG_NONFINITE_CHECK_ARG="${DEBUG_NONFINITE_CHECK_ARG:---debug_nonfinite_check}"
DEBUG_ABORT_ON_NONFINITE_ARG="${DEBUG_ABORT_ON_NONFINITE_ARG:---no-debug_abort_on_nonfinite}"
DEBUG_GRAD_TOPK="${DEBUG_GRAD_TOPK:-8}"
DEBUG_PARAM_PATTERNS="${DEBUG_PARAM_PATTERNS:-module.encoder_q.fc,encoder_q.fc,module.encoder_q.layer4,encoder_q.layer4,module.encoder_q.conv1,encoder_q.conv1}"
DEBUG_PARAM_FALLBACK_LAST_N="${DEBUG_PARAM_FALLBACK_LAST_N:-4}"
DEBUG_WRITE_JSONL_ARG="${DEBUG_WRITE_JSONL_ARG:---no-debug_write_jsonl}"
DEBUG_JSONL_NAME="${DEBUG_JSONL_NAME:-debug_train_log.jsonl}"

mkdir -p "${OUT_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "============================================================"
echo "RGB J_as_test SupLoss ablation"
echo "Project root:  ${PROJECT_ROOT}"
echo "Python script: ${PY_SCRIPT}"
echo "Dataset root:  ${DATASET_ROOT}"
echo "Manifest:      ${TRAIN_MANIFEST_NAME}"
echo "Camera:        ${CAMERA_ID}"
echo "Output root:   ${OUT_ROOT}"
echo "Tiers:         ${TIER_LIST}"
echo "Ablations:     ${ABLATION_MODES}"
echo "DDP:           ${USE_DDP}"
echo "============================================================"

run_one() {
  local tier_mode="$1"
  local ablation_mode="$2"
  local run_name="$3"
  local run_dir="${OUT_ROOT}/${tier_mode}/${run_name}"

  mkdir -p "${run_dir}"

  local optional_args=()
  if [[ -n "${PREFETCH_FACTOR}" ]]; then
    optional_args+=(--prefetch_factor "${PREFETCH_FACTOR}")
  fi
  if [[ -n "${PROTO_REFRESH_PREFETCH_FACTOR}" ]]; then
    optional_args+=(--proto_refresh_prefetch_factor "${PROTO_REFRESH_PREFETCH_FACTOR}")
  fi
  if [[ -n "${NUM_PROTOTYPES_PER_CLASS}" ]]; then
    optional_args+=(--num_prototypes_per_class "${NUM_PROTOTYPES_PER_CLASS}")
  fi

  local cmd=(
    "${PYTHON_BIN}" "${PY_SCRIPT}"
    ${NO_DDP_ARG}
    --dataset_root "${DATASET_ROOT}"
    --train_manifest_name "${TRAIN_MANIFEST_NAME}"
    --label_map_json "${LABEL_MAP_JSON}"
    --weight_save_path "${run_dir}"
    --tier_mode "${tier_mode}"
    --n_frames "${N_FRAMES}"
    --rgb_mean ${RGB_MEAN}
    --rgb_std ${RGB_STD}
    --rgb_out_hw ${RGB_OUT_HW}
    --rrc_scale ${RRC_SCALE}
    --rrc_ratio ${RRC_RATIO}
    ${RGB_APPLY_SPATIAL_AUG_ARG}
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
    --rgb_blur_sigma ${RGB_BLUR_SIGMA}
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    ${PIN_MEMORY_ARG}
    ${VERIFY_PATHS_ON_INIT_ARG}
    --model_depth "${MODEL_DEPTH}"
    --proj_dim "${PROJ_DIM}"
    --K_queue "${K_QUEUE}"
    --temperature "${TEMPERATURE}"
    ${USE_MLP_ARG}
    --contrastive_loss "${CONTRASTIVE_LOSS}"
    --num_positive "${NUM_POSITIVE}"
    ${EXCLUDE_INVALID_QUEUE_ARG}
    --ablation_mode "${ablation_mode}"
    --warmup_epochs "${WARMUP_EPOCHS}"
    --recluster_interval "${RECLUSTER_INTERVAL}"
    --default_num_prototypes "${DEFAULT_NUM_PROTOTYPES}"
    --lambda_proto "${LAMBDA_PROTO}"
    --proto_temperature "${PROTO_TEMPERATURE}"
    ${ENABLE_PROTOTYPE_TEMPERATURE_SCALING_ARG}
    --proto_temperature_eps "${PROTO_TEMPERATURE_EPS}"
    --proto_kmeans_random_state "${PROTO_KMEANS_RANDOM_STATE}"
    --proto_kmeans_n_init "${PROTO_KMEANS_N_INIT}"
    --proto_kmeans_max_iter "${PROTO_KMEANS_MAX_ITER}"
    --proto_refresh_batch_size "${PROTO_REFRESH_BATCH_SIZE}"
    --proto_refresh_num_workers "${PROTO_REFRESH_NUM_WORKERS}"
    ${PROTO_REFRESH_PIN_MEMORY_ARG}
    ${PROTO_REFRESH_VERIFY_PATHS_ON_INIT_ARG}
    --lambda_rel "${LAMBDA_REL}"
    --proto_ema_momentum "${PROTO_EMA_MOMENTUM}"
    --preview_ema_momentum "${PREVIEW_EMA_MOMENTUM}"
    --rel_same_margin "${REL_SAME_MARGIN}"
    --rel_diff_margin "${REL_DIFF_MARGIN}"
    --rel_same_weight "${REL_SAME_WEIGHT}"
    --rel_diff_weight "${REL_DIFF_WEIGHT}"
    --rel_topk_diff_classes "${REL_TOPK_DIFF_CLASSES}"
    ${ENABLE_LOSS_STAGE_SCHEDULE_ARG}
    --proto_loss_start_epoch "${PROTO_LOSS_START_EPOCH}"
    --rel_loss_start_epoch "${REL_LOSS_START_EPOCH}"
    --rel_loss_end_epoch "${REL_LOSS_END_EPOCH}"
    --rel_lambda_schedule "${REL_LAMBDA_SCHEDULE}"
    --epochs "${EPOCHS}"
    --start_epoch "${START_EPOCH}"
    --learning_rate "${LEARNING_RATE}"
    ${USE_COSINE_LR_ARG}
    --schedule ${SCHEDULE}
    --weight_decay "${WEIGHT_DECAY}"
    --momentum "${MOMENTUM}"
    --optimizer "${OPTIMIZER}"
    --adamw_betas ${ADAMW_BETAS}
    --adamw_eps "${ADAMW_EPS}"
    ${ADAMW_AMSGRAD_ARG}
    ${SEED_ARG}
    --print_freq "${PRINT_FREQ}"
    --save_interval "${SAVE_INTERVAL}"
    ${USE_SYNCBN_ARG}
    ${FIND_UNUSED_PARAMETERS_ARG}
    ${DEBUG_MODE_ARG}
    --debug_log_interval "${DEBUG_LOG_INTERVAL}"
    ${DEBUG_GRAD_STATS_ARG}
    ${DEBUG_PARAM_UPDATE_STATS_ARG}
    ${DEBUG_BATCH_LABEL_STATS_ARG}
    ${DEBUG_PROTO_STATS_ARG}
    ${DEBUG_FEATURE_STATS_ARG}
    ${DEBUG_NONFINITE_CHECK_ARG}
    ${DEBUG_ABORT_ON_NONFINITE_ARG}
    --debug_grad_topk "${DEBUG_GRAD_TOPK}"
    --debug_param_patterns "${DEBUG_PARAM_PATTERNS}"
    --debug_param_fallback_last_n "${DEBUG_PARAM_FALLBACK_LAST_N}"
    ${DEBUG_WRITE_JSONL_ARG}
    --debug_jsonl_name "${DEBUG_JSONL_NAME}"
    "${optional_args[@]}"
  )

  echo
  echo "============================================================"
  echo "[Run] tier=${tier_mode} ablation=${ablation_mode} save=${run_dir}"
  echo "============================================================"

  if [[ "${USE_DDP}" == "1" ]]; then
    torchrun \
      --standalone \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --master_port="${MASTER_PORT}" \
      "${cmd[@]:1}"
  else
    "${cmd[@]}"
  fi
}

for tier_mode in ${TIER_LIST}; do
  for ablation_mode in ${ABLATION_MODES}; do
    case "${ablation_mode}" in
      contrastive_only)
        run_name="suploss_only_p${DEFAULT_NUM_PROTOTYPES}"
        ;;
      contrastive_rel)
        run_name="suploss_rel_p${DEFAULT_NUM_PROTOTYPES}"
        ;;
      contrastive_proto)
        run_name="suploss_proto_p${DEFAULT_NUM_PROTOTYPES}"
        ;;
      contrastive_proto_rel)
        run_name="suploss_proto_rel_p${DEFAULT_NUM_PROTOTYPES}"
        ;;
      *)
        echo "[Error] Unsupported ablation mode: ${ablation_mode}" >&2
        exit 1
        ;;
    esac
    run_one "${tier_mode}" "${ablation_mode}" "${run_name}"
  done
done

echo
echo "All runs finished successfully."
echo "Output root: ${OUT_ROOT}"
