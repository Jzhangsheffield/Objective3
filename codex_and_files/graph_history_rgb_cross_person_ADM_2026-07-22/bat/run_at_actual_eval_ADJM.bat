@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Evaluation-only retest of existing Atomic-tail Direct Fusion checkpoints.
REM Training is not repeated.  Existing test_results and completed.json files
REM are never changed.  New results are written to test_results_actual_order.
REM
REM Optional overrides before calling this script:
REM   set "ACTUAL_EVAL_PARTICIPANTS=A D M J"
REM   set "ACTUAL_EVAL_SEEDS=1 2 42"
REM   set "ACTUAL_EVAL_SCOPES=normal_only all_runs"
REM   set "ACTUAL_EVAL_POLICIES=1 10 once"
REM   set "ACTUAL_EVAL_SUMMARY_ROOT=D:\short\summary"
REM   set "ACTUAL_EVAL_DRY_RUN=1"

call "%~dp0config_windows.bat" || exit /b 1
cd /d "%PACKAGE_ROOT%" || exit /b 1

if not defined ACTUAL_EVAL_PARTICIPANTS set "ACTUAL_EVAL_PARTICIPANTS=A D M J"
if not defined ACTUAL_EVAL_SEEDS set "ACTUAL_EVAL_SEEDS=1 2 42"
if not defined ACTUAL_EVAL_SCOPES set "ACTUAL_EVAL_SCOPES=normal_only all_runs"
if not defined ACTUAL_EVAL_POLICIES set "ACTUAL_EVAL_POLICIES=1 10 once"
if not defined ACTUAL_EVAL_SUMMARY_ROOT set "ACTUAL_EVAL_SUMMARY_ROOT=%OUTPUTS_ROOT%\at_actual"
if not defined ACTUAL_EVAL_DRY_RUN set "ACTUAL_EVAL_DRY_RUN=0"

if not exist "%PYTHON_BIN%" if exist "%USERPROFILE%\anaconda3\envs\Pytorch\python.exe" set "PYTHON_BIN=%USERPROFILE%\anaconda3\envs\Pytorch\python.exe"
if not exist "%PYTHON_BIN%" if exist "%USERPROFILE%\miniconda3\envs\pytorch\python.exe" set "PYTHON_BIN=%USERPROFILE%\miniconda3\envs\pytorch\python.exe"
if not "%ACTUAL_EVAL_DRY_RUN%"=="1" if not exist "%PYTHON_BIN%" (
  echo [ERROR] Configured Python executable does not exist:
  echo         %PYTHON_BIN%
  echo Set PYTHON_BIN before calling this script, for example:
  echo   set "PYTHON_BIN=C:\path\to\python.exe"
  exit /b 1
)

set "ATOMIC_SHORT_ROOT=%OUTPUTS_ROOT%\at_ad"
set "ACTUAL_EVAL_POLICY_LABELS="
for %%R in (%ACTUAL_EVAL_POLICIES%) do (
  set "CURRENT_REFRESH_LABEL=refresh_every_%%R"
  if /I "%%R"=="once" set "CURRENT_REFRESH_LABEL=refresh_once"
  set "ACTUAL_EVAL_POLICY_LABELS=!ACTUAL_EVAL_POLICY_LABELS! !CURRENT_REFRESH_LABEL!"
)

echo ============================================================
echo Atomic-tail Direct Fusion / actual-order evaluation
echo Participants: %ACTUAL_EVAL_PARTICIPANTS%
echo Seeds:        %ACTUAL_EVAL_SEEDS%
echo Scopes:       %ACTUAL_EVAL_SCOPES%
echo Policies:     %ACTUAL_EVAL_POLICIES%
echo Atomic root:  %ATOMIC_SHORT_ROOT%
echo Summary root: %ACTUAL_EVAL_SUMMARY_ROOT%
echo Dry run:      %ACTUAL_EVAL_DRY_RUN%
echo ============================================================

for %%P in (%ACTUAL_EVAL_PARTICIPANTS%) do (
  for %%S in (%ACTUAL_EVAL_SEEDS%) do (
    for %%C in (%ACTUAL_EVAL_SCOPES%) do (
      for %%R in (%ACTUAL_EVAL_POLICIES%) do (
        call :evaluate_one %%P %%S %%C %%R
        if errorlevel 1 (
          echo [ERROR] Actual-order evaluation stopped.
          exit /b 1
        )
      )
    )
  )
)

if "%ACTUAL_EVAL_DRY_RUN%"=="1" (
  echo.
  echo [DRY RUN COMPLETE] All requested checkpoint, config, cache, and manifest roots were checked.
  exit /b 0
)

echo.
echo ============================================================
echo Building paired Atomic actual-order vs M2 Direct summary
echo ============================================================
if exist "%ACTUAL_EVAL_SUMMARY_ROOT%\completed.json" (
  echo [SKIP] Completed summary already exists:
  echo        %ACTUAL_EVAL_SUMMARY_ROOT%
) else (
  "%PYTHON_BIN%" tools\summarize_atomic_tail_actual_order.py ^
    --outputs-root "%OUTPUTS_ROOT%" ^
    --atomic-root "%ATOMIC_SHORT_ROOT%" ^
    --output-dir "%ACTUAL_EVAL_SUMMARY_ROOT%" ^
    --camera-id "%CAMERA_ID%" ^
    --participants %ACTUAL_EVAL_PARTICIPANTS% ^
    --seeds %ACTUAL_EVAL_SEEDS% ^
    --train-scopes %ACTUAL_EVAL_SCOPES% ^
    --refresh-policies %ACTUAL_EVAL_POLICY_LABELS%
  if errorlevel 1 (
    echo [ERROR] Summary generation failed.
    exit /b 1
  )
)

echo.
echo ============================================================
echo COMPLETED ACTUAL-ORDER EVALUATION
echo Per-checkpoint results:
echo   %ATOMIC_SHORT_ROOT%\P_sSEED\SCOPE\POLICY\
echo     m3_atomic_tail_direct_fusion\test_results_actual_order
echo Paired summary:
echo   %ACTUAL_EVAL_SUMMARY_ROOT%
echo ============================================================
exit /b 0

:evaluate_one
setlocal EnableExtensions EnableDelayedExpansion
set "CURRENT_PARTICIPANT=%~1"
set "CURRENT_SEED=%~2"
set "CURRENT_SCOPE=%~3"
set "CURRENT_REFRESH=%~4"

set "CURRENT_REFRESH_LABEL=refresh_every_!CURRENT_REFRESH!"
if /I "!CURRENT_REFRESH!"=="once" set "CURRENT_REFRESH_LABEL=refresh_once"

set "CURRENT_FOLD_ROOT=%OUTPUTS_ROOT%\!CURRENT_PARTICIPANT!_as_test\cam_%CAMERA_ID%"
set "CURRENT_PROTOCOL_ROOT=!CURRENT_FOLD_ROOT!\protocols"
set "CURRENT_RUN_ROOT=!CURRENT_FOLD_ROOT!\seed_!CURRENT_SEED!"
if /I "!CURRENT_SCOPE!"=="normal_only" (
  set "CURRENT_TEST_CACHE=!CURRENT_RUN_ROOT!\features\retrained_normal_only\test_all.pt"
) else if /I "!CURRENT_SCOPE!"=="all_runs" (
  set "CURRENT_TEST_CACHE=!CURRENT_RUN_ROOT!\features\retrained_all_runs\test_all.pt"
) else (
  echo [ERROR] Unsupported train scope: !CURRENT_SCOPE!
  endlocal
  exit /b 1
)

set "CURRENT_MODEL_DIR=%ATOMIC_SHORT_ROOT%\!CURRENT_PARTICIPANT!_s!CURRENT_SEED!\!CURRENT_SCOPE!\!CURRENT_REFRESH_LABEL!\m3_atomic_tail_direct_fusion"
set "CURRENT_ACTUAL_ROOT=!CURRENT_MODEL_DIR!\test_results_actual_order"

if exist "!CURRENT_ACTUAL_ROOT!\completed.json" (
  echo [SKIP] P=!CURRENT_PARTICIPANT! seed=!CURRENT_SEED! scope=!CURRENT_SCOPE! policy=!CURRENT_REFRESH_LABEL!
  endlocal
  exit /b 0
)
if not exist "!CURRENT_MODEL_DIR!\last.pth" (
  echo [ERROR] Missing checkpoint: !CURRENT_MODEL_DIR!\last.pth
  endlocal
  exit /b 1
)
if not exist "!CURRENT_MODEL_DIR!\experiment_config.json" (
  echo [ERROR] Missing config: !CURRENT_MODEL_DIR!\experiment_config.json
  endlocal
  exit /b 1
)
if not exist "!CURRENT_TEST_CACHE!" (
  echo [ERROR] Missing test cache: !CURRENT_TEST_CACHE!
  endlocal
  exit /b 1
)
for %%X in (test_normal test_fault test_all) do (
  if not exist "!CURRENT_PROTOCOL_ROOT!\!CURRENT_SCOPE!\%%X.jsonl" (
    echo [ERROR] Missing test manifest: !CURRENT_PROTOCOL_ROOT!\!CURRENT_SCOPE!\%%X.jsonl
    endlocal
    exit /b 1
  )
)

if "%ACTUAL_EVAL_DRY_RUN%"=="1" (
  echo [DRY] P=!CURRENT_PARTICIPANT! seed=!CURRENT_SEED! scope=!CURRENT_SCOPE! policy=!CURRENT_REFRESH_LABEL!
  endlocal
  exit /b 0
)

echo ------------------------------------------------------------
echo [EVAL] P=!CURRENT_PARTICIPANT! seed=!CURRENT_SEED! scope=!CURRENT_SCOPE! policy=!CURRENT_REFRESH_LABEL!
echo ------------------------------------------------------------
"%PYTHON_BIN%" tools\evaluate_atomic_tail_actual_order.py ^
  --model-dir "!CURRENT_MODEL_DIR!" ^
  --protocol-root "!CURRENT_PROTOCOL_ROOT!" ^
  --test-cache "!CURRENT_TEST_CACHE!" ^
  --task-graph "%TASK_GRAPH%" ^
  --relation-matrix "%RELATION_MATRIX%" ^
  --train-scope "!CURRENT_SCOPE!" ^
  --output-dir "!CURRENT_ACTUAL_ROOT!" ^
  --batch-size 64 ^
  --num-workers %NUM_WORKERS%

set "EVAL_STATUS=!ERRORLEVEL!"
if not "!EVAL_STATUS!"=="0" (
  echo [ERROR] Evaluation failed with exit code !EVAL_STATUS!.
  endlocal
  exit /b 1
)
echo [DONE] !CURRENT_ACTUAL_ROOT!
endlocal
exit /b 0
