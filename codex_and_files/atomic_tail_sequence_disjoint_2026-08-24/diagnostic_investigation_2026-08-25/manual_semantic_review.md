# 增强历史人工语义审查记录

审查日期：2026-08-25  
审查范围：`manual_audit_candidates.csv` 中的 31 个候选（A1-Legacy-Once 16 个，A3-DualPos-Once 15 个）。候选按每个 held-out fold 分层抽取，覆盖最大扰动、经验最异常、经验支持较好和未发生增强四类。

## 审查口径

人工审查只读取动作标签、真实历史、增强历史、当前动作和量化指标，不读取测试顺序来指导生成。判断分四类：

1. **经验支持且语义上可接受**：新相邻转移均在过滤后的训练折出现；stage 不倒退；atomic/immediate 片段保持完整；动作语义未显示明显物理矛盾。
2. **形式合法但真实性存疑**：满足当前 graph-valid 约束，但较多启动/准备动作被远距离重排，或出现训练折未观察到的相邻转移；仅凭现有图无法确认物体、工具和设备状态是否一致。
3. **经验支持较弱**：50% 以上相邻转移在训练折未出现，或者短历史中所有新转移均未出现。它们不是数学意义上的无效顺序，但不适合作为“与真实分布等价”的强证据。
4. **未形成新样本**：重复节点触发 fallback，增强历史与真实历史相同。

## 总体人工判断

- 31 个候选中，7 个属于“经验支持且语义上可接受”，8 个属于“形式合法但真实性存疑”，8 个属于“经验支持较弱”，8 个属于“未形成新样本”。
- 没有发现 stage 倒退、atomic 线性片段被拆开、或增强操作新增图依赖冲突；这与全量结构审计一致。
- 主要问题集中在 Stage 1 的并行动作：当前图把 main switch、water pump、air compressor、extractor fan、pedal/cover 等多个操作视为可交换，但实际训练 run 只覆盖其中很少一部分相邻组合。因此“图允许”不等于“经验上常见”，更不等于“设备状态上完全等价”。
- `active_tail_only` 的实际行为需要特别说明：它以 active tail 决定是否启用增强，并保护 tail 的内部顺序；一旦启用，仍会重排 tail 之前的整个 remaining prefix，而不是只在 tail 内做小范围交换。

## 代表性案例

### 1. A1 / A fold / sample_000958：经验支持较弱

当前动作为 `grip sample from table`。真实历史的 Stage 1 准备动作较接近常见顺序；增强后把 cover 操作提前，并在 air compressor、main switch、pedal、water pump、unlock/lock 等动作之间形成多组新邻接。虽然 current target 追加后仍满足图依赖和 immediate predecessor，但 63.6% 的增强相邻转移未在训练折出现。判断：**形式合法，但经验真实性较弱**。

### 2. A1 / M fold / sample_000435：短历史中的极端经验异常

真实：`turn on main switch -> turn on water pump -> turn on air compressor`。  
增强：`turn on water pump -> turn on main switch -> turn on air compressor`。

三动作历史仍满足现有 graph-valid 约束，但两个增强相邻转移在该训练折均未出现，novel transition fraction 为 100%。判断：**图约束过宽或训练支持不足，不能把它当作高置信度真实顺序**。

### 3. A1 / J fold / sample_000438：大扰动但未破坏硬约束

真实：`main switch -> water pump -> air compressor -> extractor fan -> unlock -> put lock`。  
增强：`unlock -> put lock -> extractor fan -> air compressor -> water pump -> main switch`。

Kendall 距离 0.933，几乎反转了 startup 动作顺序；unlock/put lock 原子片段仍保持相邻，stage 和 target compatibility 均正常。判断：**结构合法，但扰动过大，物理/操作真实性需要额外状态证据**。

### 4. A1 / A fold / sample_000820：经验支持的可交换示例

真实：`unlock -> put lock -> main switch`。  
增强：`main switch -> unlock -> put lock`。

unlock/put lock 保持原子相邻，所有增强相邻转移均在训练折出现，当前动作 `turn on crimper` 的必要历史仍保留。判断：**现有证据下可接受**。

### 5. DualPos / A fold / sample_001126：tail 被保护，但前缀广泛重排

当前动作为 `grip sample from table`，active tail 以 `take plier from table` 结尾。增强保持了 tail 末端，却大幅重排了前面的设备启动、cover、pedal 和 unlock/lock 动作，Kendall 距离 0.485。判断：**说明 active-tail gating 并不等于局部小扰动；前缀真实性仍是主要风险**。

### 6. DualPos / J fold / sample_000762：较可信的长历史示例

增强重排了 Stage 1 的准备动作，但 Stage 2 的线性操作链保持完整；所有增强相邻转移均在训练折出现，novel transition fraction 为 0，current `press pedal` 的 immediate predecessor 也保持。判断：**经验支持且语义上可接受**。

### 7. DualPos / M fold / sample_000437：active tail 合法但邻接支持弱

当前动作为 `put lock on table`，增强历史末尾仍为 `unlock crimper`，满足 immediate predecessor；但增强前缀中 75% 的相邻转移未在训练折出现。判断：**target compatibility 正常，但整体历史经验真实性较弱**。

### 8. repeated-node fallback 案例：未形成增强

多个 fold 的长 Stage 2 历史包含重复动作节点，例如 `put sample on machine table`、`grip sample from machine table` 等多轮重复。当前实现检测到重复节点后直接返回真实顺序。因此这些样本在 Once 和 Every10 中都会贡献大量完全重复视图。判断：**安全但降低有效增强覆盖与多样性**。

## 审查边界

本次“人工真实性审查”是动作标签层面的语义审查，不包含原始视频、工具位姿、样品状态、故障类型和设备传感器状态。它可以发现明显的顺序异常和经验支持不足，但不能证明所有 graph-valid 顺序在真实工艺中都可执行。若要建立更强结论，应为候选顺序增加显式状态约束，或由熟悉工艺的人员对抽样序列进行盲审。
