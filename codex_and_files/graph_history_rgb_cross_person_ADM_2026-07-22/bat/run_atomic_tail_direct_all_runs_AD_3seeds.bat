@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Atomic-tail reduced grid:
REM   participants: A, D
REM   seeds:        1, 2, 42
REM   scope:        all_runs
REM   model:        m3_atomic_tail_direct_fusion
REM   refresh:      every epoch, every 10 epochs, once
REM Total: 2 participants x 3 seeds x 3 refresh policies = 18 training runs.

call "%~dp0config_windows.bat" || exit /b 1
cd /d "%PACKAGE_ROOT%" || exit /b 1

echo ============================================================
echo Atomic-tail Direct Fusion reduced experiment
echo Participants: A D
echo Seeds: 1 2 42
echo Scope: all_runs
echo Refresh policies: 1 10 once
echo Total training runs: 18
echo ============================================================

for %%P in (A D) do (
  for %%S in (1 2 42) do (
    for %%R in (1 10 once) do (
      call :run_one %%P %%S %%R
      if errorlevel 1 (
        echo Atomic-tail Direct Fusion experiment stopped after an error.
        exit /b 1
      )
    )
  )
)

echo ============================================================
echo Completed all requested Atomic-tail Direct Fusion runs.
echo Participants: A D; seeds: 1 2 42; scope: all_runs.
echo ============================================================
exit /b 0

:run_one
setlocal EnableExtensions EnableDelayedExpansion
set "CURRENT_PARTICIPANT=%~1"
set "CURRENT_SEED=%~2"
set "CURRENT_REFRESH=%~3"

set "CURRENT_FOLD_ROOT=%OUTPUTS_ROOT%\!CURRENT_PARTICIPANT!_as_test\cam_%CAMERA_ID%"
set "CURRENT_PROTOCOL_ROOT=!CURRENT_FOLD_ROOT!\protocols"
set "CURRENT_RUN_ROOT=!CURRENT_FOLD_ROOT!\seed_!CURRENT_SEED!"
set "CURRENT_FEATURE_ROOT=!CURRENT_RUN_ROOT!\features\retrained_all_runs"
set "CURRENT_ATOMIC_ROOT=!CURRENT_RUN_ROOT!\history_models\atomic_tail_graph_valid"

set "CURRENT_REFRESH_LABEL=refresh_every_!CURRENT_REFRESH!"
if /I "!CURRENT_REFRESH!"=="once" set "CURRENT_REFRESH_LABEL=refresh_once"

set "CURRENT_MODEL_ROOT=!CURRENT_ATOMIC_ROOT!\all_runs\!CURRENT_REFRESH_LABEL!\m3_atomic_tail_direct_fusion"
set "CURRENT_COMPLETED=!CURRENT_MODEL_ROOT!\completed.json"

if exist "!CURRENT_COMPLETED!" (
  echo [SKIP] participant=!CURRENT_PARTICIPANT! seed=!CURRENT_SEED! policy=!CURRENT_REFRESH_LABEL!
  echo        Completed marker already exists.
  endlocal
  exit /b 0
)

if not exist "!CURRENT_PROTOCOL_ROOT!\all_runs\train.jsonl" (
  echo [ERROR] Missing all-runs training manifest:
  echo         !CURRENT_PROTOCOL_ROOT!\all_runs\train.jsonl
  endlocal
  exit /b 1
)
if not exist "!CURRENT_FEATURE_ROOT!\train_all.pt" (
  echo [ERROR] Missing all-runs training feature cache:
  echo         !CURRENT_FEATURE_ROOT!\train_all.pt
  endlocal
  exit /b 1
)
if not exist "!CURRENT_FEATURE_ROOT!\test_all.pt" (
  echo [ERROR] Missing all-runs test feature cache:
  echo         !CURRENT_FEATURE_ROOT!\test_all.pt
  endlocal
  exit /b 1
)

echo ------------------------------------------------------------
echo [RUN] participant=!CURRENT_PARTICIPANT! seed=!CURRENT_SEED! policy=!CURRENT_REFRESH_LABEL!
echo ------------------------------------------------------------

"%PYTHON_BIN%" tools\train_atomic_tail_graph_valid.py ^
  --model m3_atomic_tail_direct_fusion ^
  --train-scope all_runs ^
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

echo [DONE] participant=!CURRENT_PARTICIPANT! seed=!CURRENT_SEED! policy=!CURRENT_REFRESH_LABEL!
endlocal
exit /b 0
