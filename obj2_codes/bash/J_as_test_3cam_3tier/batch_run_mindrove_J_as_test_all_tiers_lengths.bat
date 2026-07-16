@echo off
setlocal EnableExtensions

REM Usage:
REM   batch_run_mindrove_J_as_test_all_tiers_lengths.bat
REM   batch_run_mindrove_J_as_test_all_tiers_lengths.bat scratch
REM   batch_run_mindrove_J_as_test_all_tiers_lengths.bat full
REM   batch_run_mindrove_J_as_test_all_tiers_lengths.bat head_only
REM Optional second argument limits execution to tier1, tier2, or tier3.

set "RUN_MODE=%~1"
if "%RUN_MODE%"=="" set "RUN_MODE=scratch"
set "TIER_FILTER=%~2"
if "%TIER_FILTER%"=="" set "TIER_FILTER=all"

if /I not "%RUN_MODE%"=="scratch" if /I not "%RUN_MODE%"=="full" if /I not "%RUN_MODE%"=="head_only" (
    echo [Error] Mode must be scratch, full, or head_only. Got: %RUN_MODE%
    exit /b 2
)
if /I not "%TIER_FILTER%"=="all" if /I not "%TIER_FILTER%"=="tier1" if /I not "%TIER_FILTER%"=="tier2" if /I not "%TIER_FILTER%"=="tier3" (
    echo [Error] Tier filter must be all, tier1, tier2, or tier3. Got: %TIER_FILTER%
    exit /b 2
)

echo ============================================================
echo MindRove J_as_test batch run
echo Mode: %RUN_MODE%
echo Tier filter: %TIER_FILTER%
echo ============================================================

if /I "%TIER_FILTER%"=="all" call :run_tier tier1 14
if errorlevel 1 exit /b 1
if /I "%TIER_FILTER%"=="tier1" call :run_tier tier1 14
if errorlevel 1 exit /b 1

if /I "%TIER_FILTER%"=="all" call :run_tier tier2 27
if errorlevel 1 exit /b 1
if /I "%TIER_FILTER%"=="tier2" call :run_tier tier2 27
if errorlevel 1 exit /b 1

if /I "%TIER_FILTER%"=="all" call :run_tier tier3 31
if errorlevel 1 exit /b 1
if /I "%TIER_FILTER%"=="tier3" call :run_tier tier3 31
if errorlevel 1 exit /b 1

echo ============================================================
echo All requested MindRove runs finished successfully.
echo ============================================================
exit /b 0

:run_tier
set "CURRENT_TIER=%~1"
set "CURRENT_NUM_CLASSES=%~2"
echo.
echo [Tier] %CURRENT_TIER% num_classes=%CURRENT_NUM_CLASSES%

for %%L in (256 512 1024 2048) do (
    call "%~dp0ft_mindrove_J_as_test_common.bat" %CURRENT_TIER% %CURRENT_NUM_CLASSES% %RUN_MODE% emg %%L
    if errorlevel 1 exit /b 1
)
for %%L in (64 128 256 512) do (
    call "%~dp0ft_mindrove_J_as_test_common.bat" %CURRENT_TIER% %CURRENT_NUM_CLASSES% %RUN_MODE% imu %%L
    if errorlevel 1 exit /b 1
)
exit /b 0
