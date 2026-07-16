@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Batch-test fine-tuned J_as_test MindRove checkpoints.
REM
REM Usage:
REM   test_mindrove_J_as_test_finetuned_all_tiers_lengths.bat
REM   test_mindrove_J_as_test_finetuned_all_tiers_lengths.bat scratch
REM   test_mindrove_J_as_test_finetuned_all_tiers_lengths.bat full tier2 emg
REM   test_mindrove_J_as_test_finetuned_all_tiers_lengths.bat head_only all imu
REM
REM Arguments:
REM   1: mode filter   all / scratch / full / head_only   (default: all)
REM   2: tier filter   all / tier1 / tier2 / tier3        (default: all)
REM   3: signal filter all / emg / imu                     (default: all)
REM
REM Checkpoint selection can be overridden before calling this script, e.g.:
REM   set MATCH_BEST_VAL=0
REM   set MATCH_LAST=1
REM   call test_mindrove_J_as_test_finetuned_all_tiers_lengths.bat full tier1 emg

set "MODE_FILTER=%~1"
if not defined MODE_FILTER set "MODE_FILTER=all"
set "TIER_FILTER=%~2"
if not defined TIER_FILTER set "TIER_FILTER=all"
set "SIGNAL_FILTER=%~3"
if not defined SIGNAL_FILTER set "SIGNAL_FILTER=all"

if /I not "%MODE_FILTER%"=="all" if /I not "%MODE_FILTER%"=="scratch" if /I not "%MODE_FILTER%"=="full" if /I not "%MODE_FILTER%"=="head_only" (
    echo [Error] Mode filter must be all, scratch, full, or head_only. Got: %MODE_FILTER%
    exit /b 2
)
if /I not "%TIER_FILTER%"=="all" if /I not "%TIER_FILTER%"=="tier1" if /I not "%TIER_FILTER%"=="tier2" if /I not "%TIER_FILTER%"=="tier3" (
    echo [Error] Tier filter must be all, tier1, tier2, or tier3. Got: %TIER_FILTER%
    exit /b 2
)
if /I not "%SIGNAL_FILTER%"=="all" if /I not "%SIGNAL_FILTER%"=="emg" if /I not "%SIGNAL_FILTER%"=="imu" (
    echo [Error] Signal filter must be all, emg, or imu. Got: %SIGNAL_FILTER%
    exit /b 2
)

REM ============================================================
REM Paths
REM ============================================================
if not defined PROJECT_ROOT set "PROJECT_ROOT=D:\Junxi_data\Objective3_thermal_crimp\obj2_codes"
if not defined PY_SCRIPT set "PY_SCRIPT=%PROJECT_ROOT%\ft_and_test\train_mapstyle_finetune_and_test.py"
if not defined PYTHON_BIN set "PYTHON_BIN=C:\Users\digit\anaconda3\envs\Pytorch\python.exe"
if not defined DATASET_ROOT set "DATASET_ROOT=C:\Junxi_data_for_training_speedup\Stage_2_Mapstyle_Dataset"
if not defined LABEL_MAP_JSON set "LABEL_MAP_JSON=%DATASET_ROOT%\label_map.json"
if not defined TEST_MANIFEST set "TEST_MANIFEST=%DATASET_ROOT%\J_as_test\test_manifest.jsonl"
if not defined NORMALIZATION_STATS_JSON set "NORMALIZATION_STATS_JSON=%DATASET_ROOT%\J_as_test\mindrove_train_normalization_stats.json"
if not defined STATS_LOADER set "STATS_LOADER=%~dp0load_mindrove_stats_for_bat.py"

if not defined OUTPUT_ROOT set "OUTPUT_ROOT=%PROJECT_ROOT%\results\ft_mindrove_J_as_test_p1_seed1"
if not defined WEIGHTS_ROOT set "WEIGHTS_ROOT=%OUTPUT_ROOT%\weights"
if not defined TEST_OUTPUT_ROOT set "TEST_OUTPUT_ROOT=%OUTPUT_ROOT%\test_results"
if not defined RESULTS_CSV set "RESULTS_CSV=%TEST_OUTPUT_ROOT%\mindrove_J_as_test_finetuned_overall.csv"
if not defined LOG_ROOT set "LOG_ROOT=%TEST_OUTPUT_ROOT%\logs"
if not defined LIST_ROOT set "LIST_ROOT=%TEST_OUTPUT_ROOT%\selected_weight_lists"
if not defined TEST_SAVE_PATH set "TEST_SAVE_PATH=%TEST_OUTPUT_ROOT%\runtime"

REM ============================================================
REM Checkpoint, length, and runtime selection
REM ============================================================
if not defined MATCH_BEST_VAL set "MATCH_BEST_VAL=1"
if not defined MATCH_BEST_VAL_BALANCED set "MATCH_BEST_VAL_BALANCED=0"
if not defined MATCH_BEST_VAL_MACRO_F1 set "MATCH_BEST_VAL_MACRO_F1=0"
if not defined MATCH_LAST set "MATCH_LAST=0"

if not defined EMG_LENGTHS set "EMG_LENGTHS=256 512 1024 2048"
if not defined IMU_LENGTHS set "IMU_LENGTHS=64 128 256 512"
if not defined MAX_WEIGHTS_PER_RUN set "MAX_WEIGHTS_PER_RUN=20"
if not defined FAIL_IF_NO_WEIGHTS set "FAIL_IF_NO_WEIGHTS=0"
if not defined CLEAR_PREVIOUS_RESULTS set "CLEAR_PREVIOUS_RESULTS=1"
if not defined DRY_RUN set "DRY_RUN=0"

if not defined BATCH_SIZE set "BATCH_SIZE=64"
if not defined NUM_WORKERS_TEST set "NUM_WORKERS_TEST=8"
if not defined PREFETCH_FACTOR_TEST set "PREFETCH_FACTOR_TEST=2"
if not defined FIXED_SEED set "FIXED_SEED=1"
if not defined ENABLE_AMP_ARG set "ENABLE_AMP_ARG="

set "MODEL_DEPTH=10"
set "MINDROVE_ARCH=resnet10_1d"
set "MINDROVE_BASE_CHANNELS=64"
set "MINDROVE_STEM_KERNEL_SIZE=7"
set "MINDROVE_STEM_STRIDE=2"

if "%MATCH_BEST_VAL%%MATCH_BEST_VAL_BALANCED%%MATCH_BEST_VAL_MACRO_F1%%MATCH_LAST%"=="0000" (
    echo [Error] No checkpoint type is selected.
    exit /b 2
)
if %MAX_WEIGHTS_PER_RUN% LEQ 0 (
    echo [Error] MAX_WEIGHTS_PER_RUN must be greater than zero.
    exit /b 2
)

for %%P in ("%PROJECT_ROOT%" "%PY_SCRIPT%" "%PYTHON_BIN%" "%DATASET_ROOT%" "%LABEL_MAP_JSON%" "%TEST_MANIFEST%" "%NORMALIZATION_STATS_JSON%" "%STATS_LOADER%") do (
    if not exist "%%~P" (
        echo [Error] Required path does not exist: %%~P
        exit /b 1
    )
)

cd /d "%PROJECT_ROOT%"
if errorlevel 1 exit /b 1
set "PYTHONPATH=%PROJECT_ROOT%"

if not exist "%TEST_OUTPUT_ROOT%" mkdir "%TEST_OUTPUT_ROOT%"
if not exist "%LOG_ROOT%" mkdir "%LOG_ROOT%"
if not exist "%LIST_ROOT%" mkdir "%LIST_ROOT%"
if not exist "%TEST_SAVE_PATH%" mkdir "%TEST_SAVE_PATH%"

if /I "%CLEAR_PREVIOUS_RESULTS%"=="1" if /I not "%DRY_RUN%"=="1" (
    if exist "%RESULTS_CSV%" del /q "%RESULTS_CSV%" >nul 2>nul
)

set /a TOTAL_SELECTED=0
set /a TOTAL_CONFIGS=0

echo ============================================================
echo MindRove J_as_test fine-tuned checkpoint batch test
echo Mode filter:   %MODE_FILTER%
echo Tier filter:   %TIER_FILTER%
echo Signal filter: %SIGNAL_FILTER%
echo Weights root:  %WEIGHTS_ROOT%
echo Results CSV:   %RESULTS_CSV%
echo Dry run:       %DRY_RUN%
echo ============================================================

call :maybe_run_tier tier1 14
if errorlevel 1 exit /b 1
call :maybe_run_tier tier2 27
if errorlevel 1 exit /b 1
call :maybe_run_tier tier3 31
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo Batch test finished.
echo Configurations with selected weights: !TOTAL_CONFIGS!
echo Total selected weights:                !TOTAL_SELECTED!
echo Overall CSV:                           %RESULTS_CSV%
echo ============================================================
exit /b 0


:maybe_run_tier
set "CURRENT_TIER=%~1"
set "CURRENT_NUM_CLASSES=%~2"
if /I not "%TIER_FILTER%"=="all" if /I not "%TIER_FILTER%"=="%CURRENT_TIER%" exit /b 0

call :maybe_run_signal emg
if errorlevel 1 exit /b 1
call :maybe_run_signal imu
exit /b !ERRORLEVEL!


:maybe_run_signal
set "CURRENT_SIGNAL=%~1"
if /I not "%SIGNAL_FILTER%"=="all" if /I not "%SIGNAL_FILTER%"=="%CURRENT_SIGNAL%" exit /b 0

if /I "%CURRENT_SIGNAL%"=="emg" (
    for %%L in (%EMG_LENGTHS%) do (
        call :run_length %%L
        if errorlevel 1 exit /b 1
    )
) else (
    for %%L in (%IMU_LENGTHS%) do (
        call :run_length %%L
        if errorlevel 1 exit /b 1
    )
)
exit /b 0


:run_length
set "CURRENT_LENGTH=%~1"
call :maybe_run_mode scratch
if errorlevel 1 exit /b 1
call :maybe_run_mode full
if errorlevel 1 exit /b 1
call :maybe_run_mode head_only
exit /b !ERRORLEVEL!


:maybe_run_mode
set "CURRENT_MODE=%~1"
if /I not "%MODE_FILTER%"=="all" if /I not "%MODE_FILTER%"=="%CURRENT_MODE%" exit /b 0

if /I "%CURRENT_MODE%"=="scratch" (
    set "MODE_DIR=scratch_full"
    set "PY_FINETUNE_MODE=full"
) else (
    set "MODE_DIR=%CURRENT_MODE%"
    set "PY_FINETUNE_MODE=%CURRENT_MODE%"
)
call :run_configuration
exit /b !ERRORLEVEL!


:run_configuration
set "WEIGHT_ROOT=%WEIGHTS_ROOT%\%CURRENT_TIER%\%CURRENT_SIGNAL%\len_%CURRENT_LENGTH%\%MODE_DIR%"
set "CONFIG_TAG=%CURRENT_TIER%_%CURRENT_SIGNAL%_len_%CURRENT_LENGTH%_%CURRENT_MODE%"

if not exist "!WEIGHT_ROOT!" (
    if /I "%FAIL_IF_NO_WEIGHTS%"=="1" (
        echo [Error] Weight root does not exist: !WEIGHT_ROOT!
        exit /b 1
    )
    echo [Skip] Missing weight root: !WEIGHT_ROOT!
    exit /b 0
)

set "LEFT_SIGNAL_MEAN="
set "LEFT_SIGNAL_STD="
set "RIGHT_SIGNAL_MEAN="
set "RIGHT_SIGNAL_STD="
for /f "tokens=1,* delims==" %%A in ('%PYTHON_BIN% "%STATS_LOADER%" --json "%NORMALIZATION_STATS_JSON%" --signal !CURRENT_SIGNAL! --target-len !CURRENT_LENGTH!') do (
    set "%%A=%%B"
)
if not defined LEFT_SIGNAL_MEAN (
    echo [Error] Failed to load normalization statistics for !CONFIG_TAG!.
    exit /b 1
)

if /I "!CURRENT_SIGNAL!"=="emg" (
    set "NORM_ARGS=--mindrove_left_emg_mean !LEFT_SIGNAL_MEAN! --mindrove_left_emg_std !LEFT_SIGNAL_STD! --mindrove_right_emg_mean !RIGHT_SIGNAL_MEAN! --mindrove_right_emg_std !RIGHT_SIGNAL_STD!"
) else (
    set "NORM_ARGS=--mindrove_left_imu_mean !LEFT_SIGNAL_MEAN! --mindrove_left_imu_std !LEFT_SIGNAL_STD! --mindrove_right_imu_mean !RIGHT_SIGNAL_MEAN! --mindrove_right_imu_std !RIGHT_SIGNAL_STD!"
)

set "LIST_FILE=%LIST_ROOT%\!CONFIG_TAG!_weights.txt"
if exist "!LIST_FILE!" del /q "!LIST_FILE!" >nul 2>nul
set /a CONFIG_WEIGHT_COUNT=0
set /a PART_ID=1
set /a PART_COUNT=0
set "PART_WEIGHTS="

echo.
echo [Scan] !CONFIG_TAG!
echo [Root] !WEIGHT_ROOT!

if /I "%MATCH_BEST_VAL%"=="1" call :collect_checkpoint best_val.pth
if errorlevel 1 exit /b 1
if /I "%MATCH_BEST_VAL_BALANCED%"=="1" call :collect_checkpoint best_val_balanced.pth
if errorlevel 1 exit /b 1
if /I "%MATCH_BEST_VAL_MACRO_F1%"=="1" call :collect_checkpoint best_val_macro_f1.pth
if errorlevel 1 exit /b 1
if /I "%MATCH_LAST%"=="1" call :collect_checkpoint last.pth
if errorlevel 1 exit /b 1

if defined PART_WEIGHTS (
    call :run_python_part
    if errorlevel 1 exit /b 1
)

if !CONFIG_WEIGHT_COUNT! EQU 0 (
    if /I "%FAIL_IF_NO_WEIGHTS%"=="1" (
        echo [Error] No selected checkpoints found under: !WEIGHT_ROOT!
        exit /b 1
    )
    echo [Skip] No selected checkpoints found for !CONFIG_TAG!.
    exit /b 0
)

set /a TOTAL_CONFIGS+=1
echo [Done] !CONFIG_TAG! selected=!CONFIG_WEIGHT_COUNT!
exit /b 0


:collect_checkpoint
set "CKPT_NAME=%~1"
for /f "usebackq delims=" %%F in (`powershell -NoProfile -Command "Get-ChildItem -LiteralPath '!WEIGHT_ROOT!' -Recurse -Filter '!CKPT_NAME!' -File ^| Sort-Object FullName ^| ForEach-Object { $_.FullName }"`) do (
    if exist "%%~fF" (
        set /a CONFIG_WEIGHT_COUNT+=1
        set /a TOTAL_SELECTED+=1
        set /a PART_COUNT+=1
        echo %%~fF>>"!LIST_FILE!"
        if defined PART_WEIGHTS (
            set "PART_WEIGHTS=!PART_WEIGHTS! "%%~fF""
        ) else (
            set "PART_WEIGHTS="%%~fF""
        )
        if !PART_COUNT! GEQ %MAX_WEIGHTS_PER_RUN% (
            call :run_python_part
            if errorlevel 1 exit /b 1
            set /a PART_ID+=1
            set /a PART_COUNT=0
            set "PART_WEIGHTS="
        )
    )
)
exit /b 0


:run_python_part
set "CONSOLE_LOG=%LOG_ROOT%\!CONFIG_TAG!_part_!PART_ID!.log"
echo [Test] !CONFIG_TAG! part=!PART_ID! weights=!PART_COUNT!

if /I "%DRY_RUN%"=="1" (
    echo [DryRun] "%PYTHON_BIN%" "%PY_SCRIPT%" --run_mode test --tier_mode !CURRENT_TIER! --mindrove_signals !CURRENT_SIGNAL! --mindrove_target_len !CURRENT_LENGTH! --test_weight_paths !PART_WEIGHTS!
    exit /b 0
)

"%PYTHON_BIN%" "%PY_SCRIPT%" ^
  --run_mode test ^
  --dataset_root "%DATASET_ROOT%" ^
  --label_map_json "%LABEL_MAP_JSON%" ^
  --test_manifest "%TEST_MANIFEST%" ^
  --test_weight_paths !PART_WEIGHTS! ^
  --test_results_csv "%RESULTS_CSV%" ^
  --save_path "%TEST_SAVE_PATH%" ^
  --use_modality mindrove ^
  --tier_mode !CURRENT_TIER! ^
  --num_classes !CURRENT_NUM_CLASSES! ^
  --batch_size %BATCH_SIZE% ^
  --seed %FIXED_SEED% ^
  --num_workers_test %NUM_WORKERS_TEST% ^
  --prefetch_factor_test %PREFETCH_FACTOR_TEST% ^
  --model_depth %MODEL_DEPTH% ^
  --finetune_mode !PY_FINETUNE_MODE! ^
  --mindrove_hands left right ^
  --mindrove_signals !CURRENT_SIGNAL! ^
  --mindrove_target_len !CURRENT_LENGTH! ^
  --mindrove_apply_normalization ^
  --no-mindrove_apply_augmentation ^
  !NORM_ARGS! ^
  --mindrove_arch %MINDROVE_ARCH% ^
  --mindrove_base_channels %MINDROVE_BASE_CHANNELS% ^
  --mindrove_stem_kernel_size %MINDROVE_STEM_KERNEL_SIZE% ^
  --mindrove_stem_stride %MINDROVE_STEM_STRIDE% ^
  --mindrove_use_stem_pool ^
  --no-mindrove_zero_init_residual ^
  %ENABLE_AMP_ARG% ^
  > "!CONSOLE_LOG!" 2^>^&1

if errorlevel 1 (
    echo [Error] Test failed for !CONFIG_TAG! part=!PART_ID!.
    echo         See: !CONSOLE_LOG!
    exit /b 1
)
exit /b 0
