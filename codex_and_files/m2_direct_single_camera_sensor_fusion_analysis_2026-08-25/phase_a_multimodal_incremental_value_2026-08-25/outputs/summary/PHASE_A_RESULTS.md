# Phase A 实验结果汇总

状态：PENDING。

本页由 `tools/summarize_phase_a.py` 自动生成；不做 best-seed 选择。

## 完整性

- 已找到的 fold×seed×split 指标文件：54
- 尚缺指标文件：234

## 主要输出

- `condition_summary.csv`：总体、Normal、Fault。
- `per_stage.csv`：Stage 分层。
- `per_31_tier3.csv`：31 Tier3 全类别。
- `per_35_node.csv`：35 node 全类别。
- `top_12_confusions.csv`：每条件当前前 12 个 node 与 Tier3 混淆对。
- `paired_bootstrap_Ax_vs_A0.json`：配对 clip bootstrap CI。
- `incremental_value_gates.json`：多数正增益、弱类、Fault、压力和延迟门槛。

## 尚未运行

实验代码包已经就绪；以下结果需要完成 GPU 任务后生成。
首批缺失示例：

- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\A1\A_as_test\seed_2\test_results\test_all_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\A1\A_as_test\seed_2\test_results\test_normal_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\A1\A_as_test\seed_2\test_results\test_fault_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\A1\A_as_test\seed_42\test_results\test_all_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\A1\A_as_test\seed_42\test_results\test_normal_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\A1\A_as_test\seed_42\test_results\test_fault_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\A1\D_as_test\seed_1\test_results\test_all_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\A1\D_as_test\seed_1\test_results\test_normal_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\A1\D_as_test\seed_1\test_results\test_fault_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\A1\D_as_test\seed_2\test_results\test_all_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\A1\D_as_test\seed_2\test_results\test_normal_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\A1\D_as_test\seed_2\test_results\test_fault_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\A1\D_as_test\seed_42\test_results\test_all_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\A1\D_as_test\seed_42\test_results\test_normal_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\A1\D_as_test\seed_42\test_results\test_fault_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\A1\J_as_test\seed_1\test_results\test_all_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\A1\J_as_test\seed_1\test_results\test_normal_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\A1\J_as_test\seed_1\test_results\test_fault_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\A1\J_as_test\seed_2\test_results\test_all_metrics.json`
- `D:\junxi_data\Objective3\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25\outputs\A1\J_as_test\seed_2\test_results\test_normal_metrics.json`
