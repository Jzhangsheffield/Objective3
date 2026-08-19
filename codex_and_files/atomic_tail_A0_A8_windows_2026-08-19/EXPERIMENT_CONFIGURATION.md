# A0–A8 实验配置说明

## 1. 研究问题与固定条件

目标是在不改变“识别活动中的未完成 atomic sequence，并将该 prefix 固定在历史尾部，再对其余历史做 task-graph-valid 重排”这一总体思路的前提下，提高跨人动作识别性能和跨 seed 稳定性。

所有实验固定：

- 输入：通过集中配置的 `shared_artifacts_root` 只读复用冻结 R3D-18 512-D clip 特征；
- 模型主干：M2-Direct single-query history attention；
- 输出：35 个 task-graph node 类；
- 协议：participant-level LOSO，测试折为 A、D、J、M；
- seeds：1、2、42；
- 默认 scope：`all_runs`；
- 测试：`test_normal`、`test_fault`、`test_all`，一律使用真实时间历史；
- 无动作边界检测；每个已经切分的 clip 作为当前动作输入；
- 防泄漏：重排函数只接收当前 clip 之前的 history，不接收当前 target 或 future clip。

## 2. 共同模型

当前 clip 特征为 `x ∈ R^512`。历史 clip 特征先投影到 256 维，与 position embedding 相加；当前 clip 作为单 query，通过 4-head attention 得到 history context `h ∈ R^256`。`[x; h]` 经线性融合回 512 维，再由 35-node 分类头输出 logits。融合层初始化为“保留当前特征、历史分量为零”，避免训练开始时随机 history 直接破坏视觉表征。

主要监督始终为 node cross-entropy。A8 另加入 Tier-3 聚合损失，但测试指标仍同时报告 node 和 Tier-3。

## 3. Atomic-tail 判定

给定真实历史节点序列 `H`：

1. 取最后一个真实历史节点；
2. 查找包含它的 atomic sequence；
3. 若该 sequence 在 `H` 中出现的节点恰好形成从首节点开始的 proper prefix，且 sequence 尚未完成，则判为 active tail；
4. 检查将 prefix 固定在尾部不会违反其余已观察节点的必须前置约束；
5. 重复节点、非 prefix、完整 sequence 或约束冲突均回退。

这个规则不查看当前动作标签。`augmentation_audit.json` 中的 `tail_reason_counts` 可以审核每种回退原因。

## 4. 各实验详细设置

### A0 — Actual-order baseline

目的：提供与所有增强实验严格配对的 M2-Direct 基线。

- 训练 view：actual chronological history；
- position ID：真实顺序下的距离，最近历史为 1；
- 默认：读取同 participant、seed、scope 的共享旧 `m2_direct\last.pth`，进行兼容名称映射后直接评估，不复制权重；
- 可选：将 `reuse_shared_a0_checkpoint=false` 后随机初始化并训练 `baseline_epochs=50`，AdamW，LR `1e-3`，weight decay `1e-4`；
- 测试：actual；
- 主要比较：所有 Ai − A0 的 participant/seed paired delta。

### A1 — Legacy atomic-tail once

目的：在新代码中复现旧 atomic-tail once 行为，作为方法起点。

- 若存在 active tail：固定 tail，对其余历史均匀 graph-valid shuffle；
- 若不存在 active tail：仍对整个历史执行普通 graph-valid shuffle；
- 一个样本在整个训练中固定一个 shuffle（`refresh_interval="once"`）；
- position ID 按重排后位置重新编号；
- 单 augmented view，从头训练 50 epoch；
- 测试仍改为 actual，以排除测试增强带来的混杂。

解释：A1 的“无 tail 也 shuffle”会使方法退化为部分 M3，因此 A1 不是最终建议，仅用于复现与诊断。

### A2 — Active-tail-only gating

目的：减少错误或无关增强。

- 仅 `active_incomplete_atomic_prefix` 样本可重排；
- 无 active tail 的样本完全保持 actual；
- sampling 仍为均匀 graph-valid；
- position ID 仍按呈现顺序编号；
- 单 augmented view，从头训练 50 epoch。

预期：相对 A1，提高 train/test 分布一致性，减少对约 30% 真正发生变化样本之外数据的无意义扰动。该百分比应以各折新生成的 audit 为准。

### A3 — Preserve true recency

目的：把“事件身份”和“真实时间距离”随事件一起移动，避免给模型注入错误 recency。

- 继承 A2；
- 对历史元素 `h_i`，position ID 始终是它在 actual history 中距当前 clip 的真实距离；
- shuffle 只改变 attention 的呈现顺序，不伪造事件发生时间；
- 从头训练 50 epoch。

A2 与 A3 的唯一区别是 position ID 语义，因此 A3−A2 可直接检验位置错配是否是旧方法的不稳定来源。

### A4 — Paired warm-start and calibration

目的：把增强从“替代训练分布”改为“局部正则化”。

- 继承 A3；
- 默认直接加载同 participant、seed、scope 的共享旧 M2-Direct `last.pth`；若关闭共享 A0，则加载新包本地 A0 `last.pth`；
- 每个 batch 同时前向 actual 和 augmented，两者 node CE 等权平均；
- mixed fine-tuning：20 epoch，LR `3e-4`；
- 最后 actual-only calibration：5 epoch，LR `1e-4`；
- 测试 actual。

这一设置借鉴了“同一样本的多个增强视图共同训练、并控制增强强度”的一般思想。AugMix 展示了多视图与一致性正则对增强鲁棒性的价值；这里不使用其图像混合操作，只采用“保留原视图”的训练原则。[AugMix, ICLR 2020](https://openreview.net/pdf?id=S1gmrxHFvB)

注意：A4 相比 A3 同时引入 paired view、A0 warm-start 和 calibration，是一个面向性能的组合阶段。若需发表级机制拆分，可在配置复制出 A4a/A4b 补充实验，但不要改变主 A0–A8 编号。

### A5 — Confidence-gated consistency

目的：要求 actual 与合法 atomic shuffle 对同一当前 clip 给出稳定预测。

- 继承 A4；
- 对 actual logits 的最大概率 ≥ `0.7` 的样本，加入 symmetric KL；
- actual 分支概率在一致性项中 stop-gradient，避免两个分支相互追逐；
- 一致性权重 `0.2`；
- node CE 仍是主损失。

Mean Teacher 奠定了模型预测一致性作为半监督/正则化信号的常用形式；Time-Equivariant Contrastive Learning 强调视频表示应对时间变换保持可预测关系。本实验只借用一致性原则，并没有复现其网络或训练协议。[Mean Teacher, NeurIPS 2017](https://proceedings.nips.cc/paper_files/paper/2017/file/68053af2923e00204c3ca7c6a3150cf7-Paper.pdf)；[Time-Equivariant Contrastive Video Representation Learning, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Jenni_Time-Equivariant_Contrastive_Video_Representation_Learning_ICCV_2021_paper.html)

建议额外扫：consistency weight `{0.05, 0.1, 0.2, 0.4}`，threshold `{0.6, 0.7, 0.8}`。只允许用训练内验证选择。

### A6 — Plausibility-weighted constrained sampling

目的：不再从所有合法拓扑序中等概率采样，而优先产生“合法且接近真实工艺分布”的历史。

- 继承 A5；
- 仅用外层训练 fold 的完整 run 估计一阶 node transition counts；
- Laplace smoothing `0.5`；
- 每个样本生成最多 16 个 graph-valid candidate；
- candidate 的 normalized Kendall distance 必须 ≤ `0.35`；
- 至少 2 个位置发生改变；
- 保留最近 1 个 non-tail 历史节点作为锚点；
- candidate 得分为相邻转移 log probability 之和，以 temperature `0.75` 加权抽样；
- 没有合格 candidate 时安全回退 actual。

Video-Mined Task Graphs 将步骤转移表示为概率图而非简单均匀排列，支持“合法性之外还需建模过程常见度”的动机。本实现是面向本数据的轻量训练折先验，并非复制论文算法。[Video-Mined Task Graphs for Keystep Recognition, NeurIPS 2023](https://arxiv.org/abs/2307.08763)

A6 的首要审核字段是：changed fraction、平均 Kendall distance、各折收益。如果 changed fraction 过低，可依次将 candidate count 增至 32、锚点降为 0、max distance 增至 0.45；不要首先取消 active-tail gating。

### A7 — Tail-order auxiliary discrimination

目的：让 history encoder 显式学习 atomic prefix 的局部顺序，而不是只靠当前动作 CE 间接学习。

- 继承 A6；
- 对长度 ≥2 的 valid active tail，交换尾部最后两个 atomic 元素，构造 corrupted history；
- 用共享 history encoder 的 context 进行 valid=1 / corrupted=0 二分类；
- 辅助权重 `0.15`；
- corrupted view 只在训练中使用，不参与 node CE，不进入测试。

Pairwise Order Consistency、Action Shuffle Alternating Learning 与 CORP 都说明顺序判别或打乱恢复可以形成有效的程序视频监督信号。本实现把这一思想限制在已知 atomic tail 内，避免改变总体方法。[Pairwise Order Consistency, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Lu_Set-Supervised_Action_Learning_in_Procedural_Task_Videos_via_Pairwise_Order_CVPR_2022_paper.html)；[Action Shuffle Alternating Learning, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Action_Shuffle_Alternating_Learning_for_Unsupervised_Action_Segmentation_CVPR_2021_paper.html)；[Contrast and Order Representations, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Hu_Contrast_and_Order_Representations_for_Video_Self-Supervised_Learning_ICCV_2021_paper.html)

若 `tail_aux_eligible_fraction` 很低，A7 不显著并不等于顺序监督无效；应同时报告有效样本数。可在不改变方法的前提下增加“相邻 atomic pair 顺序判断”，但必须另做补充消融。

### A8 — Full method with Tier-3 auxiliary loss

目的：在 35-node 细粒度监督之外，用 31 类 Tier-3 语义约束同义/映射节点的总概率。

- 继承 A7；
- 先对 node softmax 概率按 `node_to_tier3` 做可微 `scatter_add`；
- 对真实 Tier-3 概率计算 NLL；
- 权重 `0.2`；
- paired 阶段对 actual 与 augmented 两个分支取平均；
- actual calibration 阶段保留该辅助项；
- 测试无需额外 head，Tier-3 仍由 node 概率聚合得到。

A8−A7 用于判断层级标签是否提供稳定增益。若 node accuracy 下降而 Tier-3 上升，应报告两者并将权重扫为 `{0.05, 0.1, 0.2}`，不能只选择更好看的指标。

## 5. 损失函数

主损失：

`L_node = CE(y_node, z)`

A4+ paired 阶段：

`L_pair = 0.5 CE(y, z_actual) + 0.5 CE(y, z_aug)`

A5+：

`L = L_pair + λ_cons L_symKL`

A7+：

`L = L_pair + λ_cons L_symKL + λ_order L_BCE`

A8：

`L = L_pair + λ_cons L_symKL + λ_order L_BCE + λ_tier3 L_tier3`

默认权重分别为 `0.2 / 0.15 / 0.2`，均在集中配置文件中修改。

## 6. 结果判定

“明显提升”不应只看 12 次运行的平均 accuracy。至少报告：

- `test_all` node accuracy、macro-F1；
- Tier-3 accuracy、macro-F1；
- participant × seed paired delta vs A0；
- win/tie/loss；
- 各 participant 均值和最差 participant；
- augmentation changed fraction、Kendall distance、A7 eligible fraction；
- 对 run 级预测进行 paired bootstrap 95% CI，避免把 clip 当作完全独立样本。

主张方法有效的建议门槛：平均 node accuracy 比 A0 高至少 1 个百分点、12 个配对中至少 9 胜、且最差 participant 不明显退化。门槛是本项目的预注册建议，不是通用统计标准。

## 7. 推荐调参顺序

1. 固定模型和训练预算，完成 A0–A3；
2. 若 A3 优于 A2，固定 true-recency；否则检查 position ID 与 cache 对齐；
3. A4 验证 paired + warm-start + calibration；
4. A5 只调 consistency weight/threshold；
5. A6 再调 candidate count、distance、anchor、temperature；
6. A7/A8 最后加入，避免多损失同时搜索；
7. 选择配置后完整跑 4 folds × 3 seeds，一次性冻结结果。

## 8. 集中配置字段速查

- `paths.*`：所有 Python、输入、资产、输出路径模板；
- `paths.shared_artifacts_root`：共享特征和既有 M2-Direct 权重的唯一根目录；
- `grid.*`：participants、seeds、scope、测试 split、依赖和跳过策略；
- `model.*`：模型维度；
- `training.*`：epoch、batch、worker、LR、AMP 和 device；
- `training.reuse_shared_a0_checkpoint`：`true` 时 A0 和 A4–A8 只读复用共享 M2-Direct 权重；
- `augmentation.*`：采样与辅助损失超参数；
- `experiments.A0...A8`：每个实验的不可歧义开关。

路径模板支持：`{package_root}`、`{input_root}`、`{participant}`、`{seed}`、`{scope}`、`{cache_scope}`、`{camera_id}`。

## 9. 参考文献

1. Lu et al. “Set-Supervised Action Learning in Procedural Task Videos via Pairwise Order Consistency.” CVPR 2022.
2. Li et al. “Action Shuffle Alternating Learning for Unsupervised Action Segmentation.” CVPR 2021.
3. Hu et al. “Contrast and Order Representations for Video Self-Supervised Learning.” ICCV 2021.
4. Jenni et al. “Time-Equivariant Contrastive Video Representation Learning.” ICCV 2021.
5. Wang et al. “Video-Mined Task Graphs for Keystep Recognition.” NeurIPS 2023.
6. Hendrycks et al. “AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty.” ICLR 2020.
7. Tarvainen and Valpola. “Mean Teachers Are Better Role Models.” NeurIPS 2017.
