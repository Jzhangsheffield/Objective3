@echo off
REM Central Windows path configuration. Override any variable before calling a script.
for %%I in ("%~dp0..") do if not defined PACKAGE_ROOT set "PACKAGE_ROOT=%%~fI"
if not defined DATASET_ROOT set "DATASET_ROOT=C:\MyFolder\mes19jz\Stage_2_Mapstyle_Dataset"
if not defined OBJ2_ROOT set "OBJ2_ROOT=D:\junxi_data\Objective3\obj2_codes"
if not defined EXISTING_BACKBONE set "EXISTING_BACKBONE=%OBJ2_ROOT%\results\ft_rgb_J_as_test_cam001484412812_p1_26runs_seed1\scratch_full\run_01_scratch\last.pth"
if not defined PYTHON_BIN set "PYTHON_BIN=C:\Users\mes19jz\AppData\Local\miniconda3\envs\pytorch\python.exe"
if not defined CAMERA_ID set "CAMERA_ID=001484412812"
if not defined OUTPUT_ROOT set "OUTPUT_ROOT=%PACKAGE_ROOT%\outputs\J_as_test\cam_%CAMERA_ID%"
if not defined PROTOCOL_ROOT set "PROTOCOL_ROOT=%OUTPUT_ROOT%\protocols"
if not defined FEATURE_ROOT set "FEATURE_ROOT=%OUTPUT_ROOT%\features\existing_last"
if not defined MODEL_ROOT set "MODEL_ROOT=%OUTPUT_ROOT%\history_models\existing_last_seed3"
if not defined TASK_GRAPH set "TASK_GRAPH=%PACKAGE_ROOT%\assets\integrated_task_graph_latest.json"
if not defined RELATION_MATRIX set "RELATION_MATRIX=%PACKAGE_ROOT%\assets\integrated_feature_history_matrix.json"
if not defined SEED set "SEED=3"
if not defined NUM_WORKERS set "NUM_WORKERS=8"
set "PYTHONPATH=%PACKAGE_ROOT%;%PYTHONPATH%"

