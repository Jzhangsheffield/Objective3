@echo off
setlocal EnableExtensions
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
if not exist "%ALLRUN_FEATURE_ROOT%\completed.json" (
  echo Missing completed all-runs feature cache: %ALLRUN_FEATURE_ROOT%
  exit /b 1
)
for %%M in (m0 m1 m2 m3 m4 m5 m6) do (
  call :run_model %%M
  if errorlevel 1 exit /b 1
)
exit /b 0

:run_model
if exist "%ALLRUN_MODEL_ROOT%\all_runs\%~1\completed.json" (
  echo Completed %~1 all-runs model exists; skip without overwrite.
  exit /b 0
)
echo ==== Training %~1 / complete all-runs pipeline ====
"%PYTHON_BIN%" tools\train_history_model.py ^
  --model %~1 ^
  --train-scope all_runs ^
  --protocol-root "%PROTOCOL_ROOT%" ^
  --train-cache "%ALLRUN_FEATURE_ROOT%\train_all.pt" ^
  --test-cache "%ALLRUN_FEATURE_ROOT%\test_all.pt" ^
  --task-graph "%TASK_GRAPH%" ^
  --relation-matrix "%RELATION_MATRIX%" ^
  --output-root "%ALLRUN_MODEL_ROOT%" ^
  --epochs %HISTORY_EPOCHS% ^
  --batch-size 64 ^
  --num-workers %NUM_WORKERS% ^
  --seed %SEED%
exit /b %ERRORLEVEL%
