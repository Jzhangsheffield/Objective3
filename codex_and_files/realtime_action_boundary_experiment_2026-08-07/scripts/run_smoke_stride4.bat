@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Single-condition smoke launcher: heldout=A, seed=1, scope=all_runs.
rem Usage:
rem   run_smoke_stride4.bat STAGE [CONFIG] [PYTHON_EXE]

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
if not exist "%ROOT%\config_windows.bat" (
    echo [ERROR] Missing central config: %ROOT%\config_windows.bat
    exit /b 2
)
set "PACKAGE_ROOT=%ROOT%"
call "%ROOT%\config_windows.bat"
if errorlevel 1 exit /b !errorlevel!

set "STAGE=%~1"
set "CONFIG=%~2"
set "PYTHON_EXE=%~3"

if not defined STAGE goto :usage
if not defined CONFIG set "CONFIG=%SMOKE_CONFIG%"
if not defined PYTHON_EXE set "PYTHON_EXE=%PYTHON_BIN%"

set "HELDOUT=%SMOKE_HELDOUT%"
set "SEED=%SMOKE_SEED%"
set "TRAIN_SCOPE=%SMOKE_SCOPE%"
set "ONLINE_RUN=%SMOKE_ONLINE_RUN%"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python executable not found:
    echo         %PYTHON_EXE%
    echo Pass it as the third argument.
    exit /b 2
)
if not exist "%CONFIG%" (
    echo [ERROR] Config file not found:
    echo         %CONFIG%
    exit /b 2
)

if /I "%STAGE%"=="all" (
    call :run_stage validate || exit /b !errorlevel!
    call :run_stage prepare || exit /b !errorlevel!
    call :run_stage extract || exit /b !errorlevel!
    call :run_stage train || exit /b !errorlevel!
    call :run_stage evaluate || exit /b !errorlevel!
    call :run_stage online || exit /b !errorlevel!
    echo.
    echo [DONE] Complete stride-4 smoke pipeline finished.
    exit /b 0
)

if /I "%STAGE%"=="validate" goto :single
if /I "%STAGE%"=="prepare" goto :single
if /I "%STAGE%"=="extract" goto :single
if /I "%STAGE%"=="train" goto :single
if /I "%STAGE%"=="evaluate" goto :single
if /I "%STAGE%"=="online" goto :single
if /I "%STAGE%"=="end_to_end" goto :single
goto :usage

:single
call :run_stage %STAGE%
exit /b !errorlevel!

:run_stage
set "CURRENT_STAGE=%~1"
echo.
echo [RUN] smoke stage=!CURRENT_STAGE! heldout=%HELDOUT% seed=%SEED% scope=%TRAIN_SCOPE%
echo [TIME] started !DATE! !TIME!

if /I "!CURRENT_STAGE!"=="validate" (
    "%PYTHON_EXE%" -m unittest discover -s "%ROOT%\tests" -v || exit /b !errorlevel!
    "%PYTHON_EXE%" "%ROOT%\tools\validate_setup.py" --config "%CONFIG%" --deep
    set "RC=!errorlevel!"
    echo [TIME] finished !DATE! !TIME! exit_code=!RC!
    exit /b !RC!
)
if /I "!CURRENT_STAGE!"=="prepare" (
    "%PYTHON_EXE%" "%ROOT%\tools\prepare_protocols.py" --config "%CONFIG%"
    set "RC=!errorlevel!"
    echo [TIME] finished !DATE! !TIME! exit_code=!RC!
    exit /b !RC!
)
if /I "!CURRENT_STAGE!"=="extract" (
    "%PYTHON_EXE%" "%ROOT%\tools\extract_boundary_features.py" --config "%CONFIG%" --heldout %HELDOUT% --seed %SEED% --scope %TRAIN_SCOPE% --splits train test_all
    set "RC=!errorlevel!"
    echo [TIME] finished !DATE! !TIME! exit_code=!RC!
    exit /b !RC!
)
if /I "!CURRENT_STAGE!"=="train" (
    "%PYTHON_EXE%" "%ROOT%\tools\train_boundary.py" --config "%CONFIG%" --heldout %HELDOUT% --seed %SEED% --scope %TRAIN_SCOPE%
    set "RC=!errorlevel!"
    echo [TIME] finished !DATE! !TIME! exit_code=!RC!
    exit /b !RC!
)
if /I "!CURRENT_STAGE!"=="evaluate" (
    "%PYTHON_EXE%" "%ROOT%\tools\evaluate_boundary.py" --config "%CONFIG%" --heldout %HELDOUT% --seed %SEED% --scope %TRAIN_SCOPE%
    set "RC=!errorlevel!"
    echo [TIME] finished !DATE! !TIME! exit_code=!RC!
    exit /b !RC!
)
if /I "!CURRENT_STAGE!"=="online" (
    "%PYTHON_EXE%" "%ROOT%\tools\run_online_pipeline.py" --config "%CONFIG%" --heldout %HELDOUT% --seed %SEED% --scope %TRAIN_SCOPE% --run %ONLINE_RUN%
    set "RC=!errorlevel!"
    echo [TIME] finished !DATE! !TIME! exit_code=!RC!
    exit /b !RC!
)
if /I "!CURRENT_STAGE!"=="end_to_end" (
    "%PYTHON_EXE%" "%ROOT%\tools\evaluate_end_to_end.py" --config "%CONFIG%" --heldout %HELDOUT% --seed %SEED% --scope %TRAIN_SCOPE%
    set "RC=!errorlevel!"
    echo [TIME] finished !DATE! !TIME! exit_code=!RC!
    exit /b !RC!
)
exit /b 2

:usage
echo.
echo Usage:
echo   run_smoke_stride4.bat STAGE [CONFIG] [PYTHON_EXE]
echo.
echo STAGE:
echo   validate ^| prepare ^| extract ^| train ^| evaluate ^| online ^| end_to_end ^| all
echo.
echo Notes:
echo   all runs validate, prepare, extract, train, evaluate, and one-run online inference.
echo   end_to_end is intentionally separate because it is slower.
echo.
echo Examples:
echo   run_smoke_stride4.bat all
echo   run_smoke_stride4.bat extract
echo   run_smoke_stride4.bat train "D:\experiment\configs\smoke_stride4.json" "C:\Miniconda3\envs\boundary\python.exe"
exit /b 2
