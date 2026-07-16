@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Usage:
REM   call ft_mindrove_J_as_test_common.bat tier1 14 scratch emg 256
REM   call ft_mindrove_J_as_test_common.bat tier2 27 full imu 128
REM   call ft_mindrove_J_as_test_common.bat tier3 31 head_only emg 2048

set "TIER_MODE=%~1"
set "NUM_CLASSES=%~2"
set "RUN_MODE=%~3"
set "SIGNAL=%~4"
set "TARGET_LEN=%~5"

if "%TIER_MODE%"=="" (
    echo [Error] Missing tier argument.
    exit /b 2
)
if "%NUM_CLASSES%"=="" (
    echo [Error] Missing num_classes argument.
    exit /b 2
)
if /I not "%RUN_MODE%"=="scratch" if /I not "%RUN_MODE%"=="full" if /I not "%RUN_MODE%"=="head_only" (
    echo [Error] Mode must be scratch, full, or head_only. Got: %RUN_MODE%
    exit /b 2
)
if /I not "%SIGNAL%"=="emg" if /I not "%SIGNAL%"=="imu" (
    echo [Error] Signal must be emg or imu. Got: %SIGNAL%
    exit /b 2
)
if "%TARGET_LEN%"=="" (
    echo [Error] Missing target length.
    exit /b 2
)
if /I "%SIGNAL%"=="emg" (
    set "VALID_LENGTH=0"
    for %%L in (256 512 1024 2048) do if "%TARGET_LEN%"=="%%L" set "VALID_LENGTH=1"
    if "!VALID_LENGTH!"=="0" (
        echo [Error] EMG target length must be 256, 512, 1024, or 2048. Got: %TARGET_LEN%
        exit /b 2
    )
)
if /I "%SIGNAL%"=="imu" (
    set "VALID_LENGTH=0"
    for %%L in (64 128 256 512) do if "%TARGET_LEN%"=="%%L" set "VALID_LENGTH=1"
    if "!VALID_LENGTH!"=="0" (
        echo [Error] IMU target length must be 64, 128, 256, or 512. Got: %TARGET_LEN%
        exit /b 2
    )
)

REM ============================================================
REM 1) Project, data, helper, and result paths
REM ============================================================
if not defined PROJECT_ROOT set "PROJECT_ROOT=D:\Junxi_data\Objective3_thermal_crimp\obj2_codes"
if not defined PY_SCRIPT set "PY_SCRIPT=%PROJECT_ROOT%\ft_and_test\train_mapstyle_finetune_and_test.py"
if not defined DATASET_ROOT set "DATASET_ROOT=C:\Junxi_data_for_training_speedup\Stage_2_Mapstyle_Dataset"
if not defined LABEL_MAP_JSON set "LABEL_MAP_JSON=%DATASET_ROOT%\label_map.json"
if not defined TRAIN_MANIFEST set "TRAIN_MANIFEST=%DATASET_ROOT%\J_as_test\train_manifest.jsonl"
if not defined VAL_MANIFEST set "VAL_MANIFEST=%DATASET_ROOT%\J_as_test\test_manifest.jsonl"
if not defined NORMALIZATION_STATS_JSON set "NORMALIZATION_STATS_JSON=%DATASET_ROOT%\J_as_test\mindrove_train_normalization_stats.json"

if not defined STATS_LOADER set "STATS_LOADER=%~dp0load_mindrove_stats_for_bat.py"
if not defined PYTHON_BIN set "PYTHON_BIN=C:\Users\digit\anaconda3\envs\Pytorch\python.exe"

REM Expected default pretraining layout:
REM   PRETRAIN_PARENT\weights\tier1\emg\len_256\...\checkpoint_0200.pth
REM Override PRETRAIN_ROOT or PRETRAIN_PARENT if the actual layout differs.
if not defined PRETRAIN_PARENT set "PRETRAIN_PARENT=%PROJECT_ROOT%\results\cl_mindrove_J_as_test_p1_26runs"
if not defined PRETRAIN_ROOT set "PRETRAIN_ROOT=%PRETRAIN_PARENT%\weights\%TIER_MODE%\%SIGNAL%\len_%TARGET_LEN%"

if not defined OUTPUT_ROOT set "OUTPUT_ROOT=%PROJECT_ROOT%\results\ft_mindrove_J_as_test_p1_seed1"
set "SAVE_ROOT=%OUTPUT_ROOT%\weights\%TIER_MODE%\%SIGNAL%\len_%TARGET_LEN%"
set "DATAMAP_ROOT=%OUTPUT_ROOT%\datamaps\%TIER_MODE%\%SIGNAL%\len_%TARGET_LEN%"
set "CKPT_LIST_ROOT=%OUTPUT_ROOT%\checkpoint_lists\%TIER_MODE%\%SIGNAL%\len_%TARGET_LEN%"

if not defined CHECKPOINT_NAME set "CHECKPOINT_NAME=checkpoint_0200.pth"
if not defined EXPECTED_NUM_CKPTS set "EXPECTED_NUM_CKPTS=26"
if not defined ALLOW_CKPT_COUNT_MISMATCH set "ALLOW_CKPT_COUNT_MISMATCH=0"
if not defined DRY_RUN set "DRY_RUN=0"

REM ============================================================
REM 2) Load target-length-specific J_as_test normalization stats
REM ============================================================
for %%P in ("%PROJECT_ROOT%" "%PY_SCRIPT%" "%DATASET_ROOT%" "%LABEL_MAP_JSON%" "%TRAIN_MANIFEST%" "%VAL_MANIFEST%" "%NORMALIZATION_STATS_JSON%" "%STATS_LOADER%" "%PYTHON_BIN%") do (
    if not exist "%%~P" (
        echo [Error] Required path does not exist: %%~P
        exit /b 1
    )
)

set "LEFT_SIGNAL_MEAN="
set "LEFT_SIGNAL_STD="
set "RIGHT_SIGNAL_MEAN="
set "RIGHT_SIGNAL_STD="
for /f "tokens=1,* delims==" %%A in ('%PYTHON_BIN% "%STATS_LOADER%" --json "%NORMALIZATION_STATS_JSON%" --signal %SIGNAL% --target-len %TARGET_LEN%') do (
    set "%%A=%%B"
)
if not defined LEFT_SIGNAL_MEAN (
    echo [Error] Failed to load left mean from %NORMALIZATION_STATS_JSON%
    exit /b 1
)
if not defined LEFT_SIGNAL_STD (
    echo [Error] Failed to load left std from %NORMALIZATION_STATS_JSON%
    exit /b 1
)
if not defined RIGHT_SIGNAL_MEAN (
    echo [Error] Failed to load right mean from %NORMALIZATION_STATS_JSON%
    exit /b 1
)
if not defined RIGHT_SIGNAL_STD (
    echo [Error] Failed to load right std from %NORMALIZATION_STATS_JSON%
    exit /b 1
)

set "SIGNAL_NORM_ARGS="
if /I "%SIGNAL%"=="emg" (
    set "SIGNAL_NORM_ARGS=--mindrove_left_emg_mean !LEFT_SIGNAL_MEAN! --mindrove_left_emg_std !LEFT_SIGNAL_STD! --mindrove_right_emg_mean !RIGHT_SIGNAL_MEAN! --mindrove_right_emg_std !RIGHT_SIGNAL_STD!"
)
if /I "%SIGNAL%"=="imu" (
    set "SIGNAL_NORM_ARGS=--mindrove_left_imu_mean !LEFT_SIGNAL_MEAN! --mindrove_left_imu_std !LEFT_SIGNAL_STD! --mindrove_right_imu_mean !RIGHT_SIGNAL_MEAN! --mindrove_right_imu_std !RIGHT_SIGNAL_STD!"
)

REM ============================================================
REM 3) MindRove augmentation, model, and optimizer configuration
REM ============================================================
set "USE_MODALITY=mindrove"
set "MINDROVE_HANDS=left right"
set "MINDROVE_MERGE_HANDS_ARG="
set "MINDROVE_APPLY_NORMALIZATION_ARG=--mindrove_apply_normalization"
set "MINDROVE_APPLY_AUGMENTATION_ARG=--mindrove_apply_augmentation"
set "DISABLE_TRAIN_AUGMENTATION_ARG="

if /I "%SIGNAL%"=="emg" (
    set "AUG_ARGS=--mindrove_time_warp_prob 0.5 --mindrove_time_warp_sigma 0.2 --mindrove_time_warp_num_knots 3 --mindrove_time_warp_num_splines 4 --mindrove_emg_scaling_prob 0.5 --mindrove_emg_scaling_sigma 0.10 --mindrove_emg_noise_prob 0.5 --mindrove_emg_noise_sigma 0.05 --mindrove_emg_drift_prob 0.0 --mindrove_emg_drift_max 0.2 --mindrove_emg_drift_n_points 4 --mindrove_emg_drift_kind additive --mindrove_emg_drift_per_channel --mindrove_emg_drift_normalize --mindrove_emg_negate_prob 0.0 --mindrove_emg_channel_dropout_prob 0.0 --mindrove_emg_channel_dropout_max_channels 1 --mindrove_imu_scaling_prob 0.0 --mindrove_imu_noise_prob 0.0 --mindrove_imu_drift_prob 0.0 --mindrove_imu_negate_prob 0.0 --mindrove_imu_channel_dropout_prob 0.0"
)
if /I "%SIGNAL%"=="imu" (
    set "AUG_ARGS=--mindrove_time_warp_prob 0.5 --mindrove_time_warp_sigma 0.2 --mindrove_time_warp_num_knots 3 --mindrove_time_warp_num_splines 4 --mindrove_emg_scaling_prob 0.0 --mindrove_emg_noise_prob 0.0 --mindrove_emg_drift_prob 0.0 --mindrove_emg_negate_prob 0.0 --mindrove_emg_channel_dropout_prob 0.0 --mindrove_imu_scaling_prob 0.5 --mindrove_imu_scaling_sigma 0.05 --mindrove_imu_noise_prob 0.5 --mindrove_imu_noise_sigma 0.03 --mindrove_imu_drift_prob 0.0 --mindrove_imu_drift_max 0.2 --mindrove_imu_drift_n_points 4 --mindrove_imu_drift_kind additive --mindrove_imu_drift_per_channel --mindrove_imu_drift_normalize --mindrove_imu_negate_prob 0.0 --mindrove_imu_channel_dropout_prob 0.0 --mindrove_imu_channel_dropout_max_channels 1"
)

set "BATCH_SIZE=64"
set "NUM_WORKERS_TRAIN=8"
set "NUM_WORKERS_VAL=6"
set "PREFETCH_FACTOR_TRAIN=2"
set "PREFETCH_FACTOR_VAL=2"
set "DISABLE_VAL_ARG="

set "MODEL_DEPTH=10"
set "L2_NORMALIZE_BEFORE_FC_ARG="
set "MINDROVE_ARCH=resnet10_1d"
set "MINDROVE_BASE_CHANNELS=64"
set "MINDROVE_STEM_KERNEL_SIZE=7"
set "MINDROVE_STEM_STRIDE=2"
set "MINDROVE_USE_STEM_POOL_ARG=--mindrove_use_stem_pool"
set "MINDROVE_ZERO_INIT_RESIDUAL_ARG=--no-mindrove_zero_init_residual"

set "SEED=1"
set "EPOCHS=100"
set "LEARNING_RATE=1e-3"
set "MOMENTUM=0.9"
set "WEIGHT_DECAY=1e-4"
set "OPTIMIZER=adamw"
set "ADAMW_BETA1=0.9"
set "ADAMW_BETA2=0.999"
set "ADAMW_EPS=1e-8"
set "USE_COSINE_LR_ARG="
set "SCHEDULES=50 75"
set "ENABLE_AMP_ARG="
set "SAVE_PERIOD=20"
set "BEST_AFTER_EPOCH=0"

set "HEAD_ONLY_HEAD_LR=1e-3"
set "FULL_BACKBONE_LR=1e-4"
set "FULL_HEAD_LR=1e-3"
set "USE_DISCRIMINATIVE_LR_FOR_FULL_ARG=--use_discriminative_lr"

set "KEEP_PRETRAINED_HEAD_ARG="
set "PRETRAINED_STRICT_ARG="
set "PRETRAINED_TAG_MODE=relative_to_anchor"
set "PRETRAINED_TAG_LAST_K=5"
set "PRETRAINED_TAG_ANCHOR=cl_mindrove_J_as_test_p1_26runs"

set "USE_WEIGHTED_SAMPLER_ARG="
set "SAMPLER_TIER=%TIER_MODE%"
set "SAMPLER_MODE=sqrt_inv"
set "USE_WEIGHTED_CE_ARG="
set "WEIGHT_METHOD=class_balanced"
set "CB_BETA=0.999"
set "WEIGHT_NORMALIZE_MEAN_ARG="
set "USE_FOCAL_ARG="
set "FOCAL_GAMMA=2.0"
set "FOCAL_USE_ALPHA_ARG="

cd /d "%PROJECT_ROOT%"
if errorlevel 1 exit /b 1
set "PYTHONPATH=%PROJECT_ROOT%"

if not exist "%OUTPUT_ROOT%" mkdir "%OUTPUT_ROOT%"
if not exist "%SAVE_ROOT%" mkdir "%SAVE_ROOT%"
if not exist "%DATAMAP_ROOT%" mkdir "%DATAMAP_ROOT%"
if not exist "%CKPT_LIST_ROOT%" mkdir "%CKPT_LIST_ROOT%"

echo ============================================================
echo MindRove J_as_test fine-tuning
echo Tier:          %TIER_MODE%
echo Num classes:   %NUM_CLASSES%
echo Mode:          %RUN_MODE%
echo Signal:        %SIGNAL%
echo Target length: %TARGET_LEN%
echo Output root:   %OUTPUT_ROOT%
echo Save root:     %SAVE_ROOT%
echo ============================================================

REM Scratch never checks the pretraining directory.
if /I "%RUN_MODE%"=="scratch" (
    set "SAVE_DIR=%SAVE_ROOT%\scratch_full"
    set "DATAMAP_DIR=%DATAMAP_ROOT%\scratch_full"
    if not exist "!SAVE_DIR!" mkdir "!SAVE_DIR!"
    if not exist "!DATAMAP_DIR!" mkdir "!DATAMAP_DIR!"
    call :run_python scratch_full "!SAVE_DIR!" "!DATAMAP_DIR!" full "" "%USE_DISCRIMINATIVE_LR_FOR_FULL_ARG% --backbone_learning_rate %FULL_BACKBONE_LR% --head_learning_rate %FULL_HEAD_LR%"
    exit /b !ERRORLEVEL!
)

REM full/head_only strictly require 26 matching checkpoints.
if not exist "%PRETRAIN_ROOT%" (
    echo [Error] Pretrain root does not exist: %PRETRAIN_ROOT%
    exit /b 1
)
set "WEIGHTS_PART1="
set "WEIGHTS_PART2="
set /a FOUND_COUNT=0
set /a PART1_COUNT=0
set /a PART2_COUNT=0
set "LIST_FILE=%CKPT_LIST_ROOT%\%CHECKPOINT_NAME%_paths.txt"
if exist "%LIST_FILE%" del /q "%LIST_FILE%" >nul 2>nul

for /r "%PRETRAIN_ROOT%" %%W in (%CHECKPOINT_NAME%) do (
    if exist "%%~fW" (
        set /a FOUND_COUNT+=1
        set /a PART_ID=FOUND_COUNT %% 2
        echo %%~fW>>"%LIST_FILE%"
        if !PART_ID! EQU 1 (
            set /a PART1_COUNT+=1
            if defined WEIGHTS_PART1 (set "WEIGHTS_PART1=!WEIGHTS_PART1! "%%~fW"") else (set "WEIGHTS_PART1="%%~fW"")
        ) else (
            set /a PART2_COUNT+=1
            if defined WEIGHTS_PART2 (set "WEIGHTS_PART2=!WEIGHTS_PART2! "%%~fW"") else (set "WEIGHTS_PART2="%%~fW"")
        )
    )
)

echo [Scan] %PRETRAIN_ROOT%
echo [Found] !FOUND_COUNT! %CHECKPOINT_NAME%
if !FOUND_COUNT! EQU 0 (
    echo [Error] No %CHECKPOINT_NAME% found.
    exit /b 1
)
if not "!FOUND_COUNT!"=="%EXPECTED_NUM_CKPTS%" if not "%ALLOW_CKPT_COUNT_MISMATCH%"=="1" (
    echo [Error] Expected %EXPECTED_NUM_CKPTS% checkpoints, found !FOUND_COUNT!.
    echo         Check %LIST_FILE%.
    exit /b 1
)

call :run_pretrained_part part_1
if errorlevel 1 exit /b 1
call :run_pretrained_part part_2
exit /b !ERRORLEVEL!

:run_pretrained_part
set "PART_NAME=%~1"
if /I "!PART_NAME!"=="part_1" (
    set "PART_WEIGHTS=!WEIGHTS_PART1!"
    set "PART_COUNT=!PART1_COUNT!"
) else (
    set "PART_WEIGHTS=!WEIGHTS_PART2!"
    set "PART_COUNT=!PART2_COUNT!"
)
if "!PART_WEIGHTS!"=="" exit /b 0

if /I "%RUN_MODE%"=="head_only" (
    set "FT_MODE=head_only"
    set "LR_MODE_ARGS=--head_learning_rate %HEAD_ONLY_HEAD_LR%"
) else (
    set "FT_MODE=full"
    set "LR_MODE_ARGS=%USE_DISCRIMINATIVE_LR_FOR_FULL_ARG% --backbone_learning_rate %FULL_BACKBONE_LR% --head_learning_rate %FULL_HEAD_LR%"
)
set "SAVE_DIR=%SAVE_ROOT%\!FT_MODE!\!PART_NAME!"
set "DATAMAP_DIR=%DATAMAP_ROOT%\!FT_MODE!\!PART_NAME!"
if not exist "!SAVE_DIR!" mkdir "!SAVE_DIR!"
if not exist "!DATAMAP_DIR!" mkdir "!DATAMAP_DIR!"
echo [Part] !PART_NAME! checkpoints=!PART_COUNT!
call :run_python pretrained_!FT_MODE!_!PART_NAME! "!SAVE_DIR!" "!DATAMAP_DIR!" !FT_MODE! "--pretrained_weight_paths !PART_WEIGHTS!" "!LR_MODE_ARGS!"
exit /b !ERRORLEVEL!

:run_python
set "RUN_LABEL=%~1"
set "SAVE_DIR=%~2"
set "DATAMAP_DIR=%~3"
set "FT_MODE=%~4"
set "PRETRAINED_ARGS=%~5"
set "LR_MODE_ARGS=%~6"

echo [Run] !RUN_LABEL! tier=%TIER_MODE% signal=%SIGNAL% len=%TARGET_LEN%
if /I "%DRY_RUN%"=="1" (
    echo "%PYTHON_BIN%" "%PY_SCRIPT%" --run_mode train --tier_mode %TIER_MODE% --mindrove_signals %SIGNAL% --mindrove_target_len %TARGET_LEN% --finetune_mode !FT_MODE! ...
    exit /b 0
)

"%PYTHON_BIN%" "%PY_SCRIPT%" ^
  --run_mode train ^
  --dataset_root "%DATASET_ROOT%" ^
  --label_map_json "%LABEL_MAP_JSON%" ^
  --train_manifest "%TRAIN_MANIFEST%" ^
  --val_manifest "%VAL_MANIFEST%" ^
  --save_path "!SAVE_DIR!" ^
  --datamap_csv_path "!DATAMAP_DIR!" ^
  --use_modality mindrove ^
  --tier_mode %TIER_MODE% ^
  --num_classes %NUM_CLASSES% ^
  --mindrove_hands left right ^
  --mindrove_signals %SIGNAL% ^
  --mindrove_target_len %TARGET_LEN% ^
  %MINDROVE_MERGE_HANDS_ARG% ^
  %MINDROVE_APPLY_NORMALIZATION_ARG% ^
  %MINDROVE_APPLY_AUGMENTATION_ARG% ^
  %DISABLE_TRAIN_AUGMENTATION_ARG% ^
  !SIGNAL_NORM_ARGS! ^
  !AUG_ARGS! ^
  --num_workers_train %NUM_WORKERS_TRAIN% ^
  --num_workers_val %NUM_WORKERS_VAL% ^
  --prefetch_factor_train %PREFETCH_FACTOR_TRAIN% ^
  --prefetch_factor_val %PREFETCH_FACTOR_VAL% ^
  %DISABLE_VAL_ARG% ^
  --model_depth %MODEL_DEPTH% ^
  %L2_NORMALIZE_BEFORE_FC_ARG% ^
  --mindrove_arch %MINDROVE_ARCH% ^
  --mindrove_base_channels %MINDROVE_BASE_CHANNELS% ^
  --mindrove_stem_kernel_size %MINDROVE_STEM_KERNEL_SIZE% ^
  --mindrove_stem_stride %MINDROVE_STEM_STRIDE% ^
  %MINDROVE_USE_STEM_POOL_ARG% ^
  %MINDROVE_ZERO_INIT_RESIDUAL_ARG% ^
  --epochs %EPOCHS% ^
  --batch_size %BATCH_SIZE% ^
  --learning_rate %LEARNING_RATE% ^
  --momentum %MOMENTUM% ^
  --weight_decay %WEIGHT_DECAY% ^
  --optimizer %OPTIMIZER% ^
  --adamw_beta1 %ADAMW_BETA1% ^
  --adamw_beta2 %ADAMW_BETA2% ^
  --adamw_eps %ADAMW_EPS% ^
  %USE_COSINE_LR_ARG% ^
  --schedules %SCHEDULES% ^
  --seed %SEED% ^
  --pretrained_tag_mode %PRETRAINED_TAG_MODE% ^
  --pretrained_tag_last_k %PRETRAINED_TAG_LAST_K% ^
  --pretrained_tag_anchor %PRETRAINED_TAG_ANCHOR% ^
  %KEEP_PRETRAINED_HEAD_ARG% ^
  %PRETRAINED_STRICT_ARG% ^
  --finetune_mode !FT_MODE! ^
  !LR_MODE_ARGS! ^
  %USE_WEIGHTED_SAMPLER_ARG% ^
  --sampler_tier %SAMPLER_TIER% ^
  --sampler_mode %SAMPLER_MODE% ^
  %USE_WEIGHTED_CE_ARG% ^
  --weight_method %WEIGHT_METHOD% ^
  --cb_beta %CB_BETA% ^
  %WEIGHT_NORMALIZE_MEAN_ARG% ^
  %USE_FOCAL_ARG% ^
  --focal_gamma %FOCAL_GAMMA% ^
  %FOCAL_USE_ALPHA_ARG% ^
  %ENABLE_AMP_ARG% ^
  --save_period %SAVE_PERIOD% ^
  --best_after_epoch %BEST_AFTER_EPOCH% ^
  !PRETRAINED_ARGS!

if errorlevel 1 (
    echo [Error] Failed: !RUN_LABEL!
    exit /b 1
)
exit /b 0
