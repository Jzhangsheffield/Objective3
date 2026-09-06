@echo off
REM Central Windows configuration for the joint IMU/EMG boundary package.
REM After migration, normally only DATASET_ROOT and PYTHON_BIN need editing.

for %%I in ("%~dp0.") do if not defined PACKAGE_ROOT set "PACKAGE_ROOT=%%~fI"

REM Machine-specific inputs.
if not defined DATASET_ROOT set "DATASET_ROOT=D:\Junxi_data\MULTISENSOR_DATA_COLLECTION_Stage2_structured_data\Action_Segmentation_Dataset"
if not defined PYTHON_BIN set "PYTHON_BIN=C:\Users\digit\anaconda3\envs\Pytorch\python.exe"

REM This experiment uses the restored short-background annotation set.
if not defined ANNOTATION_ROOT set "ANNOTATION_ROOT=%DATASET_ROOT%\annotations\action_recognition_boundaries_with_background_v1"

REM Protocols are shipped inside this package. Source and destination may be the same.
if not defined PROTOCOL_ROOT set "PROTOCOL_ROOT=%PACKAGE_ROOT%\protocols"
if not defined PROTOCOL_SOURCE_ROOT set "PROTOCOL_SOURCE_ROOT=%PACKAGE_ROOT%\protocols"

REM All generated artifacts remain modality-separated.
if not defined CACHE_ROOT set "CACHE_ROOT=%PACKAGE_ROOT%\cache"
if not defined OUTPUTS_ROOT set "OUTPUTS_ROOT=%PACKAGE_ROOT%\outputs"
if not defined SMOKE_OUTPUTS_ROOT set "SMOKE_OUTPUTS_ROOT=%PACKAGE_ROOT%\outputs_smoke"
if not defined VALIDATION_ROOT set "VALIDATION_ROOT=%PACKAGE_ROOT%\validation"

REM Training and launcher defaults.
if not defined NUM_WORKERS set "NUM_WORKERS=4"
if not defined RECOMMENDED_SEEDS set "RECOMMENDED_SEEDS=1 2 42"
if not defined RECOMMENDED_PARTICIPANTS set "RECOMMENDED_PARTICIPANTS=A D J M"
if not defined RECOMMENDED_SCOPES set "RECOMMENDED_SCOPES=normal_only all_runs"
if not defined SMOKE_HELDOUT set "SMOKE_HELDOUT=A"
if not defined SMOKE_SEED set "SMOKE_SEED=1"
if not defined SMOKE_SCOPE set "SMOKE_SCOPE=all_runs"
if not defined SMOKE_ONLINE_RUN set "SMOKE_ONLINE_RUN=run_sample_000001"

REM Aliases consumed by JSON configuration files.
set "SBE_EXPERIMENT_ROOT=%PACKAGE_ROOT%"
set "SBE_DATASET_ROOT=%DATASET_ROOT%"
set "SBE_ANNOTATION_ROOT=%ANNOTATION_ROOT%"
set "SBE_PROTOCOL_SOURCE_ROOT=%PROTOCOL_SOURCE_ROOT%"
set "SBE_PROTOCOL_ROOT=%PROTOCOL_ROOT%"
set "SBE_CACHE_ROOT=%CACHE_ROOT%"
set "SBE_OUTPUTS_ROOT=%OUTPUTS_ROOT%"
set "SBE_SMOKE_OUTPUTS_ROOT=%SMOKE_OUTPUTS_ROOT%"
set "SBE_VALIDATION_ROOT=%VALIDATION_ROOT%"
set "SBE_NUM_WORKERS=%NUM_WORKERS%"
set "PYTHONPATH=%PACKAGE_ROOT%;%PYTHONPATH%"

if /I "%~1"=="show" (
    echo PACKAGE_ROOT=%PACKAGE_ROOT%
    echo DATASET_ROOT=%DATASET_ROOT%
    echo ANNOTATION_ROOT=%ANNOTATION_ROOT%
    echo PYTHON_BIN=%PYTHON_BIN%
    echo PROTOCOL_ROOT=%PROTOCOL_ROOT%
    echo CACHE_ROOT=%CACHE_ROOT%
    echo OUTPUTS_ROOT=%OUTPUTS_ROOT%
    echo SMOKE_OUTPUTS_ROOT=%SMOKE_OUTPUTS_ROOT%
    echo VALIDATION_ROOT=%VALIDATION_ROOT%
    echo NUM_WORKERS=%NUM_WORKERS%
    echo RECOMMENDED_PARTICIPANTS=%RECOMMENDED_PARTICIPANTS%
    echo RECOMMENDED_SEEDS=%RECOMMENDED_SEEDS%
    echo RECOMMENDED_SCOPES=%RECOMMENDED_SCOPES%
    echo SMOKE_HELDOUT=%SMOKE_HELDOUT%
    echo SMOKE_SEED=%SMOKE_SEED%
    echo SMOKE_SCOPE=%SMOKE_SCOPE%
    echo SMOKE_ONLINE_RUN=%SMOKE_ONLINE_RUN%
)

exit /b 0
