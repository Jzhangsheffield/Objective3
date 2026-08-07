@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Full Windows CMD launcher for the LOSO experiment grid.
rem Usage:
rem   run_loso_grid.bat STAGE [SCOPE] [CONFIG] [PYTHON_EXE]

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
if not exist "%ROOT%\config_windows.bat" (
    echo [ERROR] Missing central config: %ROOT%\config_windows.bat
    exit /b 2
)
set "PACKAGE_ROOT=%ROOT%"
call "%ROOT%\config_windows.bat"
if errorlevel 1 exit /b !errorlevel!

set "STAGE=%~1"
set "SCOPE=%~2"
set "CONFIG=%~3"
set "PYTHON_EXE=%~4"

if not defined STAGE goto :usage
if not defined SCOPE set "SCOPE=both"
if not defined CONFIG set "CONFIG=%BASE_CONFIG%"
if not defined PYTHON_EXE set "PYTHON_EXE=%PYTHON_BIN%"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python executable not found:
    echo         %PYTHON_EXE%
    echo Pass it as the fourth argument.
    exit /b 2
)
if not exist "%CONFIG%" (
    echo [ERROR] Config file not found:
    echo         %CONFIG%
    exit /b 2
)

if /I "%STAGE%"=="prepare" (
    echo [RUN] Preparing all LOSO protocols
    "%PYTHON_EXE%" "%ROOT%\tools\prepare_protocols.py" --config "%CONFIG%"
    exit /b !errorlevel!
)

if /I not "%STAGE%"=="extract" if /I not "%STAGE%"=="train" if /I not "%STAGE%"=="evaluate" if /I not "%STAGE%"=="end_to_end" goto :usage
if /I "%SCOPE%"=="both" (
    set "SCOPES=%RECOMMENDED_SCOPES%"
) else if /I "%SCOPE%"=="normal_only" (
    set "SCOPES=normal_only"
) else if /I "%SCOPE%"=="all_runs" (
    set "SCOPES=all_runs"
) else (
    echo [ERROR] Invalid scope: %SCOPE%
    goto :usage
)

for %%F in (%RECOMMENDED_PARTICIPANTS%) do (
    for %%S in (%RECOMMENDED_SEEDS%) do (
        for %%C in (!SCOPES!) do (
            call :run_one %%F %%S %%C
            if errorlevel 1 exit /b !errorlevel!
        )
    )
)

echo [DONE] Stage %STAGE% completed for scope %SCOPE%.
exit /b 0

:run_one
set "HELDOUT=%~1"
set "SEED=%~2"
set "TRAIN_SCOPE=%~3"
echo.
echo [RUN] stage=%STAGE% heldout=%HELDOUT% seed=%SEED% scope=%TRAIN_SCOPE%

if /I "%STAGE%"=="extract" (
    "%PYTHON_EXE%" "%ROOT%\tools\extract_boundary_features.py" --config "%CONFIG%" --heldout %HELDOUT% --seed %SEED% --scope %TRAIN_SCOPE% --splits train test_all
    exit /b !errorlevel!
)
if /I "%STAGE%"=="train" (
    "%PYTHON_EXE%" "%ROOT%\tools\train_boundary.py" --config "%CONFIG%" --heldout %HELDOUT% --seed %SEED% --scope %TRAIN_SCOPE%
    exit /b !errorlevel!
)
if /I "%STAGE%"=="evaluate" (
    "%PYTHON_EXE%" "%ROOT%\tools\evaluate_boundary.py" --config "%CONFIG%" --heldout %HELDOUT% --seed %SEED% --scope %TRAIN_SCOPE%
    exit /b !errorlevel!
)
if /I "%STAGE%"=="end_to_end" (
    "%PYTHON_EXE%" "%ROOT%\tools\evaluate_end_to_end.py" --config "%CONFIG%" --heldout %HELDOUT% --seed %SEED% --scope %TRAIN_SCOPE%
    exit /b !errorlevel!
)
exit /b 2

:usage
echo.
echo Usage:
echo   run_loso_grid.bat STAGE [SCOPE] [CONFIG] [PYTHON_EXE]
echo.
echo STAGE:
echo   prepare ^| extract ^| train ^| evaluate ^| end_to_end
echo.
echo SCOPE:
echo   both ^| normal_only ^| all_runs
echo   Default: both
echo.
echo Examples:
echo   run_loso_grid.bat prepare
echo   run_loso_grid.bat extract both
echo   run_loso_grid.bat train all_runs
echo   run_loso_grid.bat evaluate both "D:\experiment\configs\base.json" "C:\Miniconda3\envs\boundary\python.exe"
exit /b 2
