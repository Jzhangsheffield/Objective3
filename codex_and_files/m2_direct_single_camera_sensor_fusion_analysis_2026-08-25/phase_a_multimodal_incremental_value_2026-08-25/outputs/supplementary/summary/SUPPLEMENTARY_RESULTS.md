# 右手 EMG/IMU 补充实验汇总

状态：PENDING。

本次汇总条件：S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12。

本汇总不选择 best seed；统计单位保持四折 × 三 seed。

## 条件

- S1-S4：Tier3 预训练信号特征 → 冻结特征 → scratch M2 + Node head。
- S5-S8：独立 scratch encoder → Direct Node。
- S9-S12：独立 scratch encoder → Direct Tier3，同时作为 S1-S4 上游。

## 完整性

- 已找到 fold×seed×split 指标：36
- 尚缺指标：396

## 文件

- `condition_summary.csv`：总体、Normal、Fault。
- `per_stage.csv`：Stage 分层。
- `per_35_node.csv` / `per_31_tier3.csv`：逐类别。
- `top_12_confusions.csv`：混淆对。
- `low_recall_misclassified_samples.csv`：Recall<80% 类别的错误样本名。
- `paired_fold_seed_deltas.csv`：每个预注册比较在 12 个 fold×seed 上的逐次增益。
- `incremental_value_gates.json`：多数正增益、最弱 Recall 与 Fault 非劣门槛。
- `comparison_status.json`：预注册配对比较与 bootstrap 状态。

## 首批缺失文件

- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\A_as_test\seed_2\test_results\test_all_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\A_as_test\seed_2\test_results\test_normal_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\A_as_test\seed_2\test_results\test_fault_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\A_as_test\seed_42\test_results\test_all_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\A_as_test\seed_42\test_results\test_normal_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\A_as_test\seed_42\test_results\test_fault_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\D_as_test\seed_1\test_results\test_all_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\D_as_test\seed_1\test_results\test_normal_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\D_as_test\seed_1\test_results\test_fault_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\D_as_test\seed_2\test_results\test_all_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\D_as_test\seed_2\test_results\test_normal_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\D_as_test\seed_2\test_results\test_fault_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\D_as_test\seed_42\test_results\test_all_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\D_as_test\seed_42\test_results\test_normal_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\D_as_test\seed_42\test_results\test_fault_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\J_as_test\seed_1\test_results\test_all_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\J_as_test\seed_1\test_results\test_normal_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\J_as_test\seed_1\test_results\test_fault_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\J_as_test\seed_2\test_results\test_all_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\J_as_test\seed_2\test_results\test_normal_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\J_as_test\seed_2\test_results\test_fault_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\J_as_test\seed_42\test_results\test_all_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\J_as_test\seed_42\test_results\test_normal_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\J_as_test\seed_42\test_results\test_fault_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\M_as_test\seed_1\test_results\test_all_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\M_as_test\seed_1\test_results\test_normal_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\M_as_test\seed_1\test_results\test_fault_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\M_as_test\seed_2\test_results\test_all_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\M_as_test\seed_2\test_results\test_normal_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\supplementary\S1\M_as_test\seed_2\test_results\test_fault_metrics.json`
