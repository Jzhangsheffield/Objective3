@echo off
setlocal EnableExtensions
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
if not exist "%ALLRUN_FEATURE_ROOT%\completed.json" (
  if not exist "%ALLRUN_FEATURE_ROOT%\train_all.pt" (
    echo Missing all-runs Tier-3 train feature cache: %ALLRUN_FEATURE_ROOT%\train_all.pt
    exit /b 1
  )
  if not exist "%ALLRUN_FEATURE_ROOT%\test_all.pt" (
    echo Missing all-runs Tier-3 test feature cache: %ALLRUN_FEATURE_ROOT%\test_all.pt
    exit /b 1
  )
  echo Legacy all-runs feature cache has no completed.json; verified cache files.
)
for %%R in (%ATOMIC_TAIL_REFRESH_POLICIES%) do (
  for %%M in (
    m3_atomic_tail_frozen_m0_delta
    m3_atomic_tail_joint_head_delta
    m3_atomic_tail_direct_fusion
  ) do (
    call :run_model %%M %%R
    if errorlevel 1 exit /b 1
  )
)
exit /b 0

:run_model
set "REFRESH_LABEL=refresh_every_%~2"
if /I "%~2"=="once" set "REFRESH_LABEL=refresh_once"
if exist "%ATOMIC_TAIL_MODEL_ROOT%\all_runs\%REFRESH_LABEL%\%~1\completed.json" (
  echo Completed %~1 all-runs %REFRESH_LABEL% exists; skip without overwrite.
  exit /b 0
)
echo ==== Training %~1 / atomic-tail / all-runs / %REFRESH_LABEL% ====
if /I "%~1"=="m3_atomic_tail_frozen_m0_delta" (
  if not exist "%ALLRUN_MODEL_ROOT%\all_runs\m0\last.pth" (
    echo Missing all-runs M0 checkpoint: %ALLRUN_MODEL_ROOT%\all_runs\m0\last.pth
    exit /b 1
  )
  "%PYTHON_BIN%" tools\train_atomic_tail_graph_valid.py ^
    --model %~1 ^
    --train-scope all_runs ^
    --protocol-root "%PROTOCOL_ROOT%" ^
    --train-cache "%ALLRUN_FEATURE_ROOT%\train_all.pt" ^
    --test-cache "%ALLRUN_FEATURE_ROOT%\test_all.pt" ^
    --task-graph "%TASK_GRAPH%" ^
    --relation-matrix "%RELATION_MATRIX%" ^
    --output-root "%ATOMIC_TAIL_MODEL_ROOT%" ^
    --m0-checkpoint "%ALLRUN_MODEL_ROOT%\all_runs\m0\last.pth" ^
    --shuffle-refresh-interval %~2 ^
    --epochs %HISTORY_EPOCHS% ^
    --batch-size 64 ^
    --num-workers %NUM_WORKERS% ^
    --seed %SEED%
) else (
  "%PYTHON_BIN%" tools\train_atomic_tail_graph_valid.py ^
    --model %~1 ^
    --train-scope all_runs ^
    --protocol-root "%PROTOCOL_ROOT%" ^
    --train-cache "%ALLRUN_FEATURE_ROOT%\train_all.pt" ^
    --test-cache "%ALLRUN_FEATURE_ROOT%\test_all.pt" ^
    --task-graph "%TASK_GRAPH%" ^
    --relation-matrix "%RELATION_MATRIX%" ^
    --output-root "%ATOMIC_TAIL_MODEL_ROOT%" ^
    --shuffle-refresh-interval %~2 ^
    --epochs %HISTORY_EPOCHS% ^
    --batch-size 64 ^
    --num-workers %NUM_WORKERS% ^
    --seed %SEED%
)
exit /b %ERRORLEVEL%
