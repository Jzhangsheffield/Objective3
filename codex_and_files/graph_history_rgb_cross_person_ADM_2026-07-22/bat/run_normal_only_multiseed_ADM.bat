@echo off
setlocal EnableExtensions
for %%P in (A D M) do (
  for %%S in (2 42) do (
    call :run_fold_seed %%P %%S
    if errorlevel 1 exit /b 1
  )
)
echo Completed strict normal-only seed_2 and seed_42 for A/D/M.
exit /b 0

:run_fold_seed
setlocal
set "TEST_PARTICIPANT=%~1"
set "SEED=%~2"
for %%V in (FOLD_ROOT PROTOCOL_ROOT RUN_ROOT BACKBONE_OUTPUT BACKBONE_CKPT FEATURE_ROOT MODEL_ROOT E2E_ROOT E2E_TIER3_OUTPUT E2E_NODE_SCRATCH_OUTPUT E2E_NODE_TRANSFER_OUTPUT NORMAL_FOLD_SUMMARY_ROOT TRAIN_SCOPE_COMPARISON_FOLD_ROOT) do set "%%V="
call "%~dp0run_normal_only_complete_one_fold.bat"
exit /b %ERRORLEVEL%
