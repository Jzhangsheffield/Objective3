@echo off
setlocal EnableExtensions
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
if not exist "%FEATURE_ROOT%\completed.json" (
  echo Missing completed normal-only feature cache: %FEATURE_ROOT%
  exit /b 1
)
for %%M in (m0 m1 m2 m3 m4 m5 m6) do (
  call :run_model %%M
  if errorlevel 1 exit /b 1
)
exit /b 0

:run_model
if exist "%MODEL_ROOT%\normal_only\%~1\completed.json" (
  echo Completed %~1 normal-only model exists; skip without overwrite.
  exit /b 0
)
echo ==== Training %~1 / strict normal-only pipeline ====
"%PYTHON_BIN%" tools\train_history_model.py ^
  --model %~1 ^
  --train-scope normal_only ^
  --protocol-root "%PROTOCOL_ROOT%" ^
  --train-cache "%FEATURE_ROOT%\train_all.pt" ^
  --test-cache "%FEATURE_ROOT%\test_all.pt" ^
  --task-graph "%TASK_GRAPH%" ^
  --relation-matrix "%RELATION_MATRIX%" ^
  --output-root "%MODEL_ROOT%" ^
  --epochs %HISTORY_EPOCHS% ^
  --batch-size 64 ^
  --num-workers %NUM_WORKERS% ^
  --seed %SEED%
exit /b %ERRORLEVEL%
