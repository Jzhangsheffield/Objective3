@echo off
REM Central Windows path configuration for the realtime boundary experiment.
REM Override any variable before calling a script, or edit the defaults below.

for %%I in ("%~dp0.") do if not defined PACKAGE_ROOT set "PACKAGE_ROOT=%%~fI"

REM Machine-specific inputs. These are the main values to edit after migration.
if not defined DATASET_ROOT set "DATASET_ROOT=D:\Junxi_data\MULTISENSOR_DATA_COLLECTION_Stage2_structured_data\Action_Segmentation_Dataset"
if not defined PYTHON_BIN set "PYTHON_BIN=C:\Users\digit\anaconda3\envs\Pytorch\python.exe"
if not defined ATOMIC_PROJECT_ROOT set "ATOMIC_PROJECT_ROOT=D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\graph_history_rgb_cross_person_ADM_2026-07-22"

REM Annotation and camera fixed for this experiment version.
if not defined ANNOTATION_ROOT set "ANNOTATION_ROOT=%DATASET_ROOT%\annotations\action_recognition_boundaries_with_background_v1"
if not defined CAMERA_ID set "CAMERA_ID=001484412812"

REM New experiment outputs. Change these if cache/results should be on another disk.
if not defined PROTOCOL_ROOT set "PROTOCOL_ROOT=%PACKAGE_ROOT%\protocols"
if not defined FEATURE_CACHE_ROOT set "FEATURE_CACHE_ROOT=%PACKAGE_ROOT%\cache\features"
if not defined OUTPUTS_ROOT set "OUTPUTS_ROOT=%PACKAGE_ROOT%\outputs"
if not defined SMOKE_OUTPUTS_ROOT set "SMOKE_OUTPUTS_ROOT=%PACKAGE_ROOT%\outputs_smoke_stride4"
if not defined VALIDATION_ROOT set "VALIDATION_ROOT=%PACKAGE_ROOT%\validation"

REM Config entry points.
if not defined BASE_CONFIG set "BASE_CONFIG=%PACKAGE_ROOT%\configs\base.json"
if not defined SMOKE_CONFIG set "SMOKE_CONFIG=%PACKAGE_ROOT%\configs\smoke_stride4.json"

REM Common experiment settings and launcher defaults.
if not defined NUM_WORKERS set "NUM_WORKERS=4"
if not defined RECOMMENDED_SEEDS set "RECOMMENDED_SEEDS=1 2 42"
if not defined RECOMMENDED_PARTICIPANTS set "RECOMMENDED_PARTICIPANTS=A D J M"
if not defined RECOMMENDED_SCOPES set "RECOMMENDED_SCOPES=normal_only all_runs"
if not defined SMOKE_HELDOUT set "SMOKE_HELDOUT=A"
if not defined SMOKE_SEED set "SMOKE_SEED=1"
if not defined SMOKE_SCOPE set "SMOKE_SCOPE=all_runs"
if not defined SMOKE_ONLINE_RUN set "SMOKE_ONLINE_RUN=run_sample_000001"

REM Environment aliases consumed by JSON config expansion. Do not edit separately.
set "RAB_EXPERIMENT_ROOT=%PACKAGE_ROOT%"
set "RAB_DATASET_ROOT=%DATASET_ROOT%"
set "RAB_ANNOTATION_ROOT=%ANNOTATION_ROOT%"
set "RAB_ATOMIC_PROJECT_ROOT=%ATOMIC_PROJECT_ROOT%"
set "RAB_PROTOCOL_ROOT=%PROTOCOL_ROOT%"
set "RAB_FEATURE_CACHE_ROOT=%FEATURE_CACHE_ROOT%"
set "RAB_OUTPUTS_ROOT=%OUTPUTS_ROOT%"
set "RAB_SMOKE_OUTPUTS_ROOT=%SMOKE_OUTPUTS_ROOT%"
set "RAB_VALIDATION_ROOT=%VALIDATION_ROOT%"
set "RAB_CAMERA_ID=%CAMERA_ID%"
set "RAB_NUM_WORKERS=%NUM_WORKERS%"
set "RAB_SMOKE_HELDOUT=%SMOKE_HELDOUT%"
set "RAB_SMOKE_SEED=%SMOKE_SEED%"
set "RAB_SMOKE_SCOPE=%SMOKE_SCOPE%"

set "PYTHONPATH=%PACKAGE_ROOT%;%PYTHONPATH%"

if /I "%~1"=="show" (
    echo PACKAGE_ROOT=%PACKAGE_ROOT%
    echo DATASET_ROOT=%DATASET_ROOT%
    echo ANNOTATION_ROOT=%ANNOTATION_ROOT%
    echo PYTHON_BIN=%PYTHON_BIN%
    echo ATOMIC_PROJECT_ROOT=%ATOMIC_PROJECT_ROOT%
    echo CAMERA_ID=%CAMERA_ID%
    echo PROTOCOL_ROOT=%PROTOCOL_ROOT%
    echo FEATURE_CACHE_ROOT=%FEATURE_CACHE_ROOT%
    echo OUTPUTS_ROOT=%OUTPUTS_ROOT%
    echo SMOKE_OUTPUTS_ROOT=%SMOKE_OUTPUTS_ROOT%
    echo VALIDATION_ROOT=%VALIDATION_ROOT%
    echo BASE_CONFIG=%BASE_CONFIG%
    echo SMOKE_CONFIG=%SMOKE_CONFIG%
    echo NUM_WORKERS=%NUM_WORKERS%
    echo RECOMMENDED_SEEDS=%RECOMMENDED_SEEDS%
    echo RECOMMENDED_PARTICIPANTS=%RECOMMENDED_PARTICIPANTS%
    echo RECOMMENDED_SCOPES=%RECOMMENDED_SCOPES%
    echo SMOKE_HELDOUT=%SMOKE_HELDOUT%
    echo SMOKE_SEED=%SMOKE_SEED%
    echo SMOKE_SCOPE=%SMOKE_SCOPE%
    echo SMOKE_ONLINE_RUN=%SMOKE_ONLINE_RUN%
)

exit /b 0
