@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Usage: run_smoke.bat MODALITY STAGE [CONFIG] [PYTHON_EXE]
REM MODALITY: imu | emg | both
REM STAGE: validate | prepare | extract | train | calibrate | evaluate | online | all

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PACKAGE_ROOT=%ROOT%"
call "%ROOT%\config_windows.bat"
if errorlevel 1 exit /b !errorlevel!

set "MODALITY=%~1"
set "STAGE=%~2"
set "CONFIG=%~3"
set "PYTHON_EXE=%~4"
if not defined MODALITY goto :usage
if not defined STAGE goto :usage
if not defined PYTHON_EXE set "PYTHON_EXE=%PYTHON_BIN%"

if /I "%MODALITY%"=="both" (
    if defined CONFIG (
        echo [ERROR] A custom CONFIG can only be used with modality imu or emg, not both.
        exit /b 2
    )
    call :run_modality imu || exit /b !errorlevel!
    call :run_modality emg || exit /b !errorlevel!
    echo [DONE] smoke modality=both stage=%STAGE%
    exit /b 0
)
if /I "%MODALITY%"=="imu" goto :one
if /I "%MODALITY%"=="emg" goto :one
goto :usage

:one
call :run_modality %MODALITY%
exit /b !errorlevel!

:run_modality
set "CURRENT_MODALITY=%~1"
set "CURRENT_CONFIG=%CONFIG%"
if not defined CURRENT_CONFIG set "CURRENT_CONFIG=%ROOT%\configs\!CURRENT_MODALITY!\smoke.json"
if /I "%STAGE%"=="all" (
    call :run_stage !CURRENT_MODALITY! validate !CURRENT_CONFIG! || exit /b !errorlevel!
    call :run_stage !CURRENT_MODALITY! prepare !CURRENT_CONFIG! || exit /b !errorlevel!
    call :run_stage !CURRENT_MODALITY! extract !CURRENT_CONFIG! || exit /b !errorlevel!
    call :run_stage !CURRENT_MODALITY! train !CURRENT_CONFIG! || exit /b !errorlevel!
    call :run_stage !CURRENT_MODALITY! calibrate !CURRENT_CONFIG! || exit /b !errorlevel!
    call :run_stage !CURRENT_MODALITY! evaluate !CURRENT_CONFIG! || exit /b !errorlevel!
    call :run_stage !CURRENT_MODALITY! online !CURRENT_CONFIG! || exit /b !errorlevel!
    exit /b 0
)
call :run_stage !CURRENT_MODALITY! %STAGE% !CURRENT_CONFIG!
exit /b !errorlevel!

:run_stage
set "M=%~1"
set "S=%~2"
set "C=%~3"
echo.
echo [RUN] smoke modality=!M! stage=!S! heldout=%SMOKE_HELDOUT% seed=%SMOKE_SEED% scope=%SMOKE_SCOPE%
echo [TIME] !DATE! !TIME!
if /I "!S!"=="validate" (
    "%PYTHON_EXE%" -m unittest discover -s "%ROOT%\tests" -v || exit /b !errorlevel!
    "%PYTHON_EXE%" "%ROOT%\tools\validate_setup.py" --config "!C!"
    exit /b !errorlevel!
)
if /I "!S!"=="prepare" (
    "%PYTHON_EXE%" "%ROOT%\tools\prepare_protocols.py" --config "!C!"
    exit /b !errorlevel!
)
if /I "!S!"=="extract" (
    "%PYTHON_EXE%" "%ROOT%\tools\extract_sensor_windows.py" --config "!C!" --heldout %SMOKE_HELDOUT% --scope %SMOKE_SCOPE% --splits train test_normal test_fault test_all
    exit /b !errorlevel!
)
if /I "!S!"=="train" (
    "%PYTHON_EXE%" "%ROOT%\tools\train_boundary.py" --config "!C!" --heldout %SMOKE_HELDOUT% --seed %SMOKE_SEED% --scope %SMOKE_SCOPE% --overwrite
    exit /b !errorlevel!
)
if /I "!S!"=="calibrate" (
    "%PYTHON_EXE%" "%ROOT%\tools\calibrate_online.py" --config "!C!" --heldout %SMOKE_HELDOUT% --seed %SMOKE_SEED% --scope %SMOKE_SCOPE%
    exit /b !errorlevel!
)
if /I "!S!"=="evaluate" (
    "%PYTHON_EXE%" "%ROOT%\tools\evaluate_boundary.py" --config "!C!" --heldout %SMOKE_HELDOUT% --seed %SMOKE_SEED% --scope %SMOKE_SCOPE%
    exit /b !errorlevel!
)
if /I "!S!"=="online" (
    "%PYTHON_EXE%" "%ROOT%\tools\run_online_pipeline.py" --config "!C!" --heldout %SMOKE_HELDOUT% --seed %SMOKE_SEED% --scope %SMOKE_SCOPE% --run %SMOKE_ONLINE_RUN%
    exit /b !errorlevel!
)
echo [ERROR] Invalid stage: !S!
exit /b 2

:usage
echo Usage: run_smoke.bat MODALITY STAGE [CONFIG] [PYTHON_EXE]
echo MODALITY: imu ^| emg ^| both
echo STAGE: validate ^| prepare ^| extract ^| train ^| calibrate ^| evaluate ^| online ^| all
echo Example: scripts\run_smoke.bat both all
exit /b 2
