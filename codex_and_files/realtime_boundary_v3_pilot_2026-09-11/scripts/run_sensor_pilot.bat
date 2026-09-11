@echo off
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..") do set "PILOT_ROOT=%%~fI"
set "MODALITY=%~1"
set "STAGE=%~2"
if not defined MODALITY set "MODALITY=both"
if not defined STAGE set "STAGE=all"
if /I "%STAGE%"=="all" (
    for %%T in (validate prepare extract train evaluate summarize) do (
        call "%PILOT_ROOT%\sensor_baselines\scripts\run_loso_grid.bat" %MODALITY% %%T all_runs
        if errorlevel 1 exit /b !errorlevel!
    )
    exit /b 0
)
call "%PILOT_ROOT%\sensor_baselines\scripts\run_loso_grid.bat" %MODALITY% %STAGE% all_runs
exit /b !errorlevel!
