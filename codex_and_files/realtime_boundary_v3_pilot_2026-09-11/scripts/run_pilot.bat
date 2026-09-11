@echo off
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..") do set "PACKAGE_ROOT=%%~fI"
call "%PACKAGE_ROOT%\config_windows.bat"
set "STAGE=%~1"
if not defined STAGE set "STAGE=all"
if /I "%STAGE%"=="validate" goto :validate
if /I "%STAGE%"=="summary" goto :summary
if /I "%STAGE%"=="all" (
    call :validate
    if errorlevel 1 exit /b !errorlevel!
    for %%T in (prepare extract train evaluate end_to_end) do (
        call "%PACKAGE_ROOT%\scripts\run_loso_grid.bat" %%T all_runs
        if errorlevel 1 exit /b !errorlevel!
    )
    goto :summary
)
call "%PACKAGE_ROOT%\scripts\run_loso_grid.bat" %STAGE% all_runs
exit /b !errorlevel!
:validate
"%PYTHON_BIN%" "%PACKAGE_ROOT%\tools\validate_pilot.py"
exit /b !errorlevel!
:summary
"%PYTHON_BIN%" "%PACKAGE_ROOT%\tools\summarize_pilot.py" --outputs "%OUTPUTS_ROOT%"
exit /b !errorlevel!
