@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Atomic-tail Direct Fusion extension grid:
REM
REM Phase 1 (runs first):
REM   participants: M, J
REM   scope:        all_runs
REM
REM Phase 2 (runs only after Phase 1 completes):
REM   participants: A, D, M, J
REM   scope:        normal_only
REM
REM Both phases:
REM   seeds:        1, 2, 42
REM   model:        m3_atomic_tail_direct_fusion
REM   refresh:      every epoch, every 10 epochs, once
REM
REM Total:
REM   Phase 1: 2 participants x 3 seeds x 3 policies = 18 runs
REM   Phase 2: 4 participants x 3 seeds x 3 policies = 36 runs
REM   Overall: 54 training runs
REM
REM Keep the same short output layout used by the earlier A/D experiment:
REM   outputs\at_ad\<participant>_s<seed>\<scope>\<policy>\
REM     m3_atomic_tail_direct_fusion\
REM
REM The historical folder name "at_ad" is intentionally retained for output
REM compatibility even though this extension also contains M and J.

call "%~dp0config_windows.bat" || exit /b 1
cd /d "%PACKAGE_ROOT%" || exit /b 1

echo ============================================================
echo Atomic-tail Direct Fusion extension experiment
echo Phase 1: M J / all_runs / 18 runs
echo Phase 2: A D M J / normal_only / 36 runs
echo Seeds: 1 2 42
echo Refresh policies: 1 10 once
echo Total training runs: 54
echo Short output root: %OUTPUTS_ROOT%\at_ad
echo ============================================================

echo.
echo ============================================================
echo PHASE 1 OF 2: M and J / all_runs
echo ============================================================
for %%P in (M J) do (
  for %%S in (1 2 42) do (
    for %%R in (1 10 once) do (
      call :run_one %%P %%S %%R all_runs
      if errorlevel 1 (
        echo Phase 1 stopped after an error.
        exit /b 1
      )
    )
  )
)

echo.
echo ============================================================
echo PHASE 1 COMPLETED.
echo PHASE 2 OF 2: A, D, M and J / normal_only
echo ============================================================
for %%P in (A D M J) do (
  for %%S in (1 2 42) do (
    for %%R in (1 10 once) do (
      call :run_one %%P %%S %%R normal_only
      if errorlevel 1 (
        echo Phase 2 stopped after an error.
        exit /b 1
      )
    )
  )
)

echo.
echo ============================================================
echo COMPLETED ALL REQUESTED ATOMIC-TAIL DIRECT FUSION RUNS.
echo Phase 1: M J / all_runs
echo Phase 2: A D M J / normal_only
echo ============================================================
exit /b 0

:run_one
setlocal EnableExtensions EnableDelayedExpansion
set "CURRENT_PARTICIPANT=%~1"
set "CURRENT_SEED=%~2"
set "CURRENT_REFRESH=%~3"
set "CURRENT_SCOPE=%~4"

set "CURRENT_FOLD_ROOT=%OUTPUTS_ROOT%\!CURRENT_PARTICIPANT!_as_test\cam_%CAMERA_ID%"
set "CURRENT_PROTOCOL_ROOT=!CURRENT_FOLD_ROOT!\protocols"
set "CURRENT_RUN_ROOT=!CURRENT_FOLD_ROOT!\seed_!CURRENT_SEED!"
set "CURRENT_ATOMIC_ROOT=%OUTPUTS_ROOT%\at_ad\!CURRENT_PARTICIPANT!_s!CURRENT_SEED!"

if /I "!CURRENT_SCOPE!"=="all_runs" (
  set "CURRENT_FEATURE_ROOT=!CURRENT_RUN_ROOT!\features\retrained_all_runs"
) else if /I "!CURRENT_SCOPE!"=="normal_only" (
  set "CURRENT_FEATURE_ROOT=!CURRENT_RUN_ROOT!\features\retrained_normal_only"
) else (
  echo [ERROR] Unsupported train scope: !CURRENT_SCOPE!
  endlocal
  exit /b 1
)

set "CURRENT_REFRESH_LABEL=refresh_every_!CURRENT_REFRESH!"
if /I "!CURRENT_REFRESH!"=="once" set "CURRENT_REFRESH_LABEL=refresh_once"

set "CURRENT_MODEL_ROOT=!CURRENT_ATOMIC_ROOT!\!CURRENT_SCOPE!\!CURRENT_REFRESH_LABEL!\m3_atomic_tail_direct_fusion"
set "CURRENT_COMPLETED=!CURRENT_MODEL_ROOT!\completed.json"

if exist "!CURRENT_COMPLETED!" (
  echo [SKIP] participant=!CURRENT_PARTICIPANT! seed=!CURRENT_SEED! scope=!CURRENT_SCOPE! policy=!CURRENT_REFRESH_LABEL!
  echo        Completed marker already exists.
  endlocal
  exit /b 0
)

if not exist "!CURRENT_PROTOCOL_ROOT!\!CURRENT_SCOPE!\train.jsonl" (
  echo [ERROR] Missing !CURRENT_SCOPE! training manifest:
  echo         !CURRENT_PROTOCOL_ROOT!\!CURRENT_SCOPE!\train.jsonl
  endlocal
  exit /b 1
)
if not exist "!CURRENT_FEATURE_ROOT!\train_all.pt" (
  echo [ERROR] Missing !CURRENT_SCOPE! training feature cache:
  echo         !CURRENT_FEATURE_ROOT!\train_all.pt
  endlocal
  exit /b 1
)
if not exist "!CURRENT_FEATURE_ROOT!\test_all.pt" (
  echo [ERROR] Missing !CURRENT_SCOPE! test feature cache:
  echo         !CURRENT_FEATURE_ROOT!\test_all.pt
  endlocal
  exit /b 1
)

echo ------------------------------------------------------------
echo [RUN] participant=!CURRENT_PARTICIPANT! seed=!CURRENT_SEED! scope=!CURRENT_SCOPE! policy=!CURRENT_REFRESH_LABEL!
echo ------------------------------------------------------------

"%PYTHON_BIN%" tools\train_atomic_tail_graph_valid.py ^
  --model m3_atomic_tail_direct_fusion ^
  --train-scope "!CURRENT_SCOPE!" ^
  --protocol-root "!CURRENT_PROTOCOL_ROOT!" ^
  --train-cache "!CURRENT_FEATURE_ROOT!\train_all.pt" ^
  --test-cache "!CURRENT_FEATURE_ROOT!\test_all.pt" ^
  --task-graph "%TASK_GRAPH%" ^
  --relation-matrix "%RELATION_MATRIX%" ^
  --output-root "!CURRENT_ATOMIC_ROOT!" ^
  --shuffle-refresh-interval "!CURRENT_REFRESH!" ^
  --epochs %HISTORY_EPOCHS% ^
  --batch-size 64 ^
  --num-workers %NUM_WORKERS% ^
  --seed !CURRENT_SEED!

set "TRAIN_STATUS=!ERRORLEVEL!"
if not "!TRAIN_STATUS!"=="0" (
  echo [ERROR] Training failed with exit code !TRAIN_STATUS!.
  endlocal
  exit /b 1
)

echo [DONE] participant=!CURRENT_PARTICIPANT! seed=!CURRENT_SEED! scope=!CURRENT_SCOPE! policy=!CURRENT_REFRESH_LABEL!
endlocal
exit /b 0
