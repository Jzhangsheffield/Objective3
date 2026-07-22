@echo off
setlocal EnableExtensions
for %%P in (A D M) do (
  call :run_fold %%P
  if errorlevel 1 exit /b 1
)
call "%~dp0\12_summarize_all_models_cross_person.bat" || exit /b 1
echo Completed incremental E2E baselines for A/D/M.
exit /b 0

:run_fold
setlocal
set "TEST_PARTICIPANT=%~1"
for %%V in (FOLD_ROOT PROTOCOL_ROOT RUN_ROOT BACKBONE_OUTPUT BACKBONE_CKPT FEATURE_ROOT MODEL_ROOT E2E_ROOT E2E_TIER3_OUTPUT E2E_NODE_SCRATCH_OUTPUT E2E_NODE_TRANSFER_OUTPUT UNIFIED_FOLD_SUMMARY_ROOT) do set "%%V="
call "%~dp0run_additional_e2e_one_fold.bat"
exit /b %ERRORLEVEL%

