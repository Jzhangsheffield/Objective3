@echo off
setlocal
call "%~dp0config_windows.bat"
cd /d "%PACKAGE_ROOT%" || exit /b 1
if exist "%ALLRUN_E2E_NODE_SCRATCH_OUTPUT%\completed.json" (
  echo Completed all-runs E2E Node Scratch exists; skip without overwrite.
  exit /b 0
)
"%PYTHON_BIN%" tools\train_e2e_node.py ^
  --dataset-root "%DATASET_ROOT%" ^
  --protocol-root "%PROTOCOL_ROOT%" ^
  --task-graph "%TASK_GRAPH%" ^
  --relation-matrix "%RELATION_MATRIX%" ^
  --output-dir "%ALLRUN_E2E_NODE_SCRATCH_OUTPUT%" ^
  --init scratch ^
  --train-scope all_runs ^
  --camera-id "%CAMERA_ID%" ^
  --epochs %E2E_NODE_EPOCHS% ^
  --batch-size 16 ^
  --num-workers %NUM_WORKERS% ^
  --learning-rate %E2E_NODE_LR% ^
  --seed %SEED% ^
  --amp
exit /b %ERRORLEVEL%
