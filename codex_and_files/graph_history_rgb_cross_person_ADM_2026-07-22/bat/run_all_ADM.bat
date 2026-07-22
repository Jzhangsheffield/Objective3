@echo off
setlocal EnableExtensions
for %%P in (A D M) do (
  echo Starting %%P-as-test...
  call :run_fold %%P
  if errorlevel 1 exit /b 1
)
call "%~dp0\07_summarize_cross_person.bat" || exit /b 1
echo Completed all A/D/M folds.
exit /b 0

:run_fold
setlocal
set "TEST_PARTICIPANT=%~1"
set "FOLD_ROOT="
set "PROTOCOL_ROOT="
set "RUN_ROOT="
set "BACKBONE_OUTPUT="
set "BACKBONE_CKPT="
set "FEATURE_ROOT="
set "MODEL_ROOT="
call "%~dp0run_one_fold.bat"
exit /b %ERRORLEVEL%
