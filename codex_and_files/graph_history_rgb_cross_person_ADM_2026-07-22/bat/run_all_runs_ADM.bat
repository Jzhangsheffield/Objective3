@echo off
setlocal EnableExtensions
for %%P in (A D M) do (
  call :run_fold %%P
  if errorlevel 1 exit /b 1
)
call "%~dp0\22_summarize_all_runs_cross_person.bat" || exit /b 1
call "%~dp0\23_summarize_training_scope_comparison_cross_person.bat" || exit /b 1
echo Completed complete all-runs pipelines for A/D/M.
exit /b 0

:run_fold
setlocal
set "TEST_PARTICIPANT=%~1"
for %%V in (FOLD_ROOT PROTOCOL_ROOT RUN_ROOT ALLRUN_BACKBONE_OUTPUT ALLRUN_BACKBONE_CKPT ALLRUN_FEATURE_ROOT ALLRUN_MODEL_ROOT ALLRUN_E2E_ROOT ALLRUN_E2E_TIER3_OUTPUT ALLRUN_E2E_NODE_SCRATCH_OUTPUT ALLRUN_E2E_NODE_TRANSFER_OUTPUT ALLRUN_FOLD_SUMMARY_ROOT TRAIN_SCOPE_COMPARISON_FOLD_ROOT) do set "%%V="
call "%~dp0run_all_runs_one_fold.bat"
exit /b %ERRORLEVEL%
