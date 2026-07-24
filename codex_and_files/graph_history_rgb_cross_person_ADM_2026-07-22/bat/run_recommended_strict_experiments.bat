@echo off
setlocal EnableExtensions
call "%~dp0config_windows.bat"
for %%P in (A D M) do (
  for %%S in (1 2 42) do (
    if not exist "%OUTPUTS_ROOT%\%%P_as_test\cam_%CAMERA_ID%\seed_%%S\backbone\all_runs\completed.json" (
      echo Missing completed existing all-runs result: %%P seed=%%S
      echo Run the all-runs pipeline first or restore its outputs.
      exit /b 1
    )
  )
)
call "%~dp0run_normal_only_multiseed_ADM.bat" || exit /b 1
call "%~dp0run_strict_J_three_seeds.bat" || exit /b 1
call "%~dp0run_fourfold_ADJM_summaries.bat" || exit /b 1
echo Completed all recommended strict multiseed experiments and summaries.
exit /b 0
