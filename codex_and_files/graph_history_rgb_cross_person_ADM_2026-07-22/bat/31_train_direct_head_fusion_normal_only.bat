@echo off
setlocal EnableExtensions
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
if not exist "%FEATURE_ROOT%\completed.json" (
  if not exist "%FEATURE_ROOT%\train_all.pt" (
    echo Missing normal-only Tier-3 train feature cache: %FEATURE_ROOT%\train_all.pt
    exit /b 1
  )
  if not exist "%FEATURE_ROOT%\test_all.pt" (
    echo Missing normal-only Tier-3 test feature cache: %FEATURE_ROOT%\test_all.pt
    exit /b 1
  )
  echo Legacy normal-only feature cache has no completed.json; verified train_all.pt and test_all.pt.
)
for %%M in (m1_direct m2_direct m3_direct) do (
  call :run_model %%M
  if errorlevel 1 exit /b 1
)
exit /b 0

:run_model
if exist "%DIRECT_MODEL_ROOT%\normal_only\%~1\completed.json" (
  echo Completed %~1 normal-only direct-head model exists; skip without overwrite.
  exit /b 0
)
echo ==== Training %~1 / direct-head fusion / normal-only ====
"%PYTHON_BIN%" tools\train_direct_history_model.py ^
  --model %~1 ^
  --train-scope normal_only ^
  --protocol-root "%PROTOCOL_ROOT%" ^
  --train-cache "%FEATURE_ROOT%\train_all.pt" ^
  --test-cache "%FEATURE_ROOT%\test_all.pt" ^
  --task-graph "%TASK_GRAPH%" ^
  --relation-matrix "%RELATION_MATRIX%" ^
  --output-root "%DIRECT_MODEL_ROOT%" ^
  --epochs %HISTORY_EPOCHS% ^
  --batch-size 64 ^
  --num-workers %NUM_WORKERS% ^
  --seed %SEED%
exit /b %ERRORLEVEL%
