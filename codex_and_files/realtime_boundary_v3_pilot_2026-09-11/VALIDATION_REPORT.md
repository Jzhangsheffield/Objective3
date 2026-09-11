# 打包验证记录：2026-09-11

验证使用本机 `C:\Users\digit\anaconda3\envs\Pytorch\python.exe`。未运行正式特征提取、训练、测试集推理；以下是准备状态，不是模型性能结果。

| 检查 | 结果 |
|---|---|
| RGB 四折预检 | 103runs，251132frames，1894个GT动作，problems=[] |
| 每折GT动作 | A=431、D=462、J=555、M=446 |
| v3/node显式对应 | 全部通过数量及动作类别检查 |
| 四折RGB协议prepare | 通过 |
| IMU、EMG协议prepare | 均通过 |
| IMU深度数据检查 | 103/103通过，时间单调且与标注时间有重叠 |
| EMG深度数据检查 | 103/103通过 |
| RGB回归测试 | 9/9通过，含因果性、多worker顺序、上下文loss、状态机及相邻GT |
| 传感器回归测试 | 7/7通过，含上下文/padding loss屏蔽 |
| 四折backbone及M3 | 全部8个权重精确架构加载通过；M3空历史→非空历史合成输入检查通过 |

详细机器可读报告：`validation_canonical_v3/pilot_preflight.json`、`weight_checks.json`，以及` sensor_baselines/validation_canonical_v3/{imu|emg}/setup_validation.json`。

单调性和时间重叠通过不等于同步偏差已消除：13个run仍有来源标记的修复回退。未在新电脑验证驱动、GPU显存、吞吐、长时间训练稳定性，迁移后须先运行smoke。系统PATH中的另一个python缺少torch；所有正式bat均通过PYTHON_BIN指定解释器，勿用不明来源的python运行。
