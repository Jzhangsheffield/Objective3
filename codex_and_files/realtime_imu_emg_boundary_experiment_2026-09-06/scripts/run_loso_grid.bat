@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Usage: run_loso_grid.bat MODALITY STAGE [SCOPE] [CONFIG] [PYTHON_EXE]
REM MODALITY: imu | emg | both
REM STAGE: validate | prepare | extract | train | calibrate | evaluate | summarize

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PACKAGE_ROOT=%ROOT%"
call "%ROOT%\config_windows.bat"
if errorlevel 1 exit /b !errorlevel!

set "MODALITY=%~1"
set "STAGE=%~2"
set "SCOPE=%~3"
set "CONFIG=%~4"
set "PYTHON_EXE=%~5"
if not defined MODALITY goto :usage
if not defined STAGE goto :usage
if not defined SCOPE set "SCOPE=both"
if not defined PYTHON_EXE set "PYTHON_EXE=%PYTHON_BIN%"
if /I not "%STAGE%"=="validate" if /I not "%STAGE%"=="prepare" if /I not "%STAGE%"=="extract" if /I not "%STAGE%"=="train" if /I not "%STAGE%"=="calibrate" if /I not "%STAGE%"=="evaluate" if /I not "%STAGE%"=="summarize" goto :usage

if /I "%MODALITY%"=="both" (
    if defined CONFIG (
        echo [ERROR] A custom CONFIG can only be used with modality imu or emg, not both.
        exit /b 2
    )
    call :run_modality imu || exit /b !errorlevel!
    call :run_modality emg || exit /b !errorlevel!
    exit /b 0
)
if /I "%MODALITY%"=="imu" goto :one
if /I "%MODALITY%"=="emg" goto :one
goto :usage

:one
call :run_modality %MODALITY%
exit /b !errorlevel!

:run_modality
set "M=%~1"
set "C=%CONFIG%"
if not defined C set "C=%ROOT%\configs\!M!\base.json"
if /I "%STAGE%"=="validate" (
    "%PYTHON_EXE%" "%ROOT%\tools\validate_setup.py" --config "!C!" --deep
    exit /b !errorlevel!
)
if /I "%STAGE%"=="prepare" (
    "%PYTHON_EXE%" "%ROOT%\tools\prepare_protocols.py" --config "!C!"
    exit /b !errorlevel!
)
if /I "%STAGE%"=="summarize" (
    "%PYTHON_EXE%" "%ROOT%\tools\summarize_results.py" --outputs-root "%OUTPUTS_ROOT%" --output "%OUTPUTS_ROOT%\loso_summary.csv"
    exit /b !errorlevel!
)
if /I "%SCOPE%"=="both" (
    set "SCOPES=%RECOMMENDED_SCOPES%"
) else if /I "%SCOPE%"=="normal_only" (
    set "SCOPES=normal_only"
) else if /I "%SCOPE%"=="all_runs" (
    set "SCOPES=all_runs"
) else (
    echo [ERROR] Invalid scope: %SCOPE%
    exit /b 2
)
for %%F in (%RECOMMENDED_PARTICIPANTS%) do (
    for %%C in (!SCOPES!) do (
        if /I "%STAGE%"=="extract" (
            "%PYTHON_EXE%" "%ROOT%\tools\extract_sensor_windows.py" --config "!C!" --heldout %%F --scope %%C --splits train test_all
            if errorlevel 1 exit /b !errorlevel!
        ) else (
            for %%S in (%RECOMMENDED_SEEDS%) do (
                echo [RUN] modality=!M! stage=%STAGE% heldout=%%F seed=%%S scope=%%C
                if /I "%STAGE%"=="train" (
                    "%PYTHON_EXE%" "%ROOT%\tools\train_boundary.py" --config "!C!" --heldout %%F --seed %%S --scope %%C
                    if errorlevel 1 exit /b !errorlevel!
                )
                if /I "%STAGE%"=="calibrate" (
                    "%PYTHON_EXE%" "%ROOT%\tools\calibrate_online.py" --config "!C!" --heldout %%F --seed %%S --scope %%C
                    if errorlevel 1 exit /b !errorlevel!
                )
                if /I "%STAGE%"=="evaluate" (
                    "%PYTHON_EXE%" "%ROOT%\tools\evaluate_boundary.py" --config "!C!" --heldout %%F --seed %%S --scope %%C
                    if errorlevel 1 exit /b !errorlevel!
                )
            )
        )
    )
)
exit /b 0

:usage
echo Usage: run_loso_grid.bat MODALITY STAGE [SCOPE] [CONFIG] [PYTHON_EXE]
echo MODALITY: imu ^| emg ^| both
echo STAGE: validate ^| prepare ^| extract ^| train ^| calibrate ^| evaluate ^| summarize
echo Example: scripts\run_loso_grid.bat both extract both
exit /b 2
