@echo off
setlocal EnableExtensions
for %%P in (A D J M) do (
  for %%S in (1 2 42) do (
    call :run_seed %%P %%S
    if errorlevel 1 exit /b 1
  )
)
call "%~dp0\33_summarize_direct_head_fusion_ADJM_3seeds.bat" || exit /b 1
echo Completed direct-head fusion A/D/J/M three-seed experiment and summary.
exit /b 0

:run_seed
setlocal
set "TEST_PARTICIPANT=%~1"
set "SEED=%~2"
for %%V in (FOLD_ROOT PROTOCOL_ROOT RUN_ROOT BACKBONE_OUTPUT BACKBONE_CKPT FEATURE_ROOT MODEL_ROOT ALLRUN_BACKBONE_OUTPUT ALLRUN_BACKBONE_CKPT ALLRUN_FEATURE_ROOT ALLRUN_MODEL_ROOT DIRECT_MODEL_ROOT) do set "%%V="
call "%~dp0\run_direct_head_fusion_one_fold.bat"
exit /b %ERRORLEVEL%
