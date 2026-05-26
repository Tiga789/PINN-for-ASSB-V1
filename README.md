# PINN-for-ASSB-V1 / QJW-2

本项目基于 PINNSTRIPES / effective SPM 思路，先完成 NMC811||Li-In 全固态电池（ASSB）的五目标工程基线，再推进不同电池与不同充放电策略下的泛化能力。当前最新工作阶段为 **ASSB-D8 / GV1 数据管线阶段**。

## 当前总状态

### ASSB 基线

当前 ASSB 五目标工程统一基线仍为：

```text
ModelFin_112_deterministic_wrapper
EvalFin_112_deterministic_wrapper
```

该基线来自 D7：

- 四个电化学状态 `cs_a / cs_c / phie / phis_c` 来自 frozen `ModelFin_107A` state eval NPZ。
- SOH 来自 `ModelFin_112_deterministicSOH_ridge_g4`。
- 这是一个 engineering wrapper / unified package，不是端到端联合训练的单个神经网络。
- 它证明了当前 ASSB 本电池五目标工程封装成立，但不代表跨电池规格泛化。

### D8 / GV1 当前状态

D8 已经开始泛化能力落地，目标是：

```text
不同电池 + 不同充放电策略 + 已记录电流 I(t)
→ measured-current replay
→ CellSpec / ExperimentSpec 条件输入
→ 条件化 effective SPM PINN
```

当前已经完成 **数据通路**，尚未接入真正的条件化 PINN 网络训练。`gv1_train.py` 当前输出状态为：

```text
status = validated_not_trained
reason = PINN network/loss modules are intentionally not attached in this entry package yet.
```

D9 下一步应实现：

```text
gv1/model.py
gv1/output_transform.py
gv1/losses.py
gv1/trainer.py
scripts/gv1_train_conditioned_pinn.py
```

## GV1 设计原则

1. **不修改旧 ASSB 主线文件**  
   `main.py`、`util/*`、`integration_spm/*` 等旧主线文件不应被 GV1 改造污染。

2. **measured-current replay**  
   对恒流、分段恒流、恒压、恒功率、CC-CV 等实验，只要 record 中记录了 `I(t)`，GV1 第一版都将 `I(t)` 作为已知边界输入。

3. **不做 controller-solved**  
   当前不求解“只给恒压/恒功率设定值，模型自洽反解 I(t)”的问题。

4. **缓存大文件不放项目根目录**  
   XJTU 的 Parquet / NPZ / CSV 缓存统一放在原始数据集目录：

```text
E:/XJTU battery dataset/_gv1_cache
```

5. **训练前先审计**  
   所有数据必须先通过读取、时间轴、cycle/step、SOH 标签、profile、manifest 审计，再进入训练。

## D8 已新增的主要模块

```text
gv1/io/                 # 通用 .mat / .csv / .parquet 读取层
gv1/adapters/           # XJTU 数据集适配层
gv1/measured_replay/    # measured-current replay profile 构建
gv1/pipeline/           # manifest / npz / metrics / data_loader
cell_specs/             # 电池规格文件
experiment_specs/       # 实验策略文件
manifests/              # 数据集和训练评估清单
scripts/gv1_*.py        # GV1 数据管线入口脚本
tests/                  # GV1 smoke tests
```

## XJTU Batch-1/3/4 数据准备结果

原始数据路径：

```text
E:/XJTU battery dataset
```

当前接入的 Batch：

```text
Batch-1
Batch-3
Batch-4
```

缓存根目录：

```text
E:/XJTU battery dataset/_gv1_cache
```

### 已完成的关键结果

| 项目 | 结果 |
|---|---:|
| 原始 `.mat` 文件数 | 24 |
| 标准化 Parquet 基础审计 | 24/24 OK |
| cycle/step Parquet 文件 | 24 |
| replay profile NPZ | 24 |
| replay profile audit | 24/24 OK |
| cycle 行数 | 13513 |
| 完整放电 SOH 标签数 | 8787 |
| partial/unlabeled cycle | 4726 |
| profile split | train 18 / val 3 / test 3 |
| SOH labeled split | train 6594 / val 1030 / test 1163 |
| training-ready linked rows | 13513 / 13513 |

## 关键缓存目录

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_standard
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_cycle_step_standard
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_cycle_summary
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_soh_labels
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_training_manifest
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_replay_profiles
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_replay_profile_audit
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_training_ready
```

## 当前最重要的 JSON / CSV

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_soh_labels/xjtu_batch134_soh_label_table.csv
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_soh_labels/xjtu_batch134_soh_label_report.json

E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_training_manifest/xjtu_batch134_cycle_manifest.csv
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_training_manifest/xjtu_batch134_cycle_manifest_labeled_only.csv

E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_replay_profile_audit/replay_profile_audit_summary.json

E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_training_ready/xjtu_batch134_profile_manifest.csv
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_training_ready/xjtu_batch134_cycle_training_manifest.csv
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_training_ready/xjtu_batch134_training_ready_report.json
```

## 重要技术修复记录

### 1. 时间轴修复

Batch-3 / Batch-4 的 `.mat` 直接用 `system_time` 拼接会导致 `time_s is not monotonic nondecreasing`。

当前使用：

```text
GV1 time fix v3
```

处理方式：

```text
raw__mat_subrecord_index + local elapsed time → monotonic global time_s
```

### 2. cycle/step 恢复

XJTU 标准表中 `cycle_id` 初始为空，但 `raw__mat_subrecord_index` 完整存在。

当前约定：

```text
cycle_id = raw__mat_subrecord_index + 1
step_id  = cycle 内连续 step_type/current_A 段编号
```

### 3. SOH 标签

SOH 不使用 `solution_replay_profile.npz` 内的 `Q_ref_Ah_replay`。当前 SOH 标签来自：

```text
SOH = q_discharge_Ah / mean(first_3_full_discharge_q_discharge_Ah_per_cell)
```

只有完整放电 cycle 作为 SOH 标签；Batch-4 的部分放电 cycle 保留用于 replay，但不作为 SOH 真值。

### 4. streaming 生成 profile

一次性读取 24 个多百万行 Parquet 会导致内存问题。当前 `gv1_generate_softlabels.py` 已改为逐文件 streaming 生成 profile。

## 当前训练入口状态

训练入口 smoke 输出：

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_train_smoke_1profile/training_plan.json
```

状态：

```text
validated_not_trained
```

原因：GV1 的条件化 PINN 网络、output transform、loss 和 trainer 尚未实现。

评估入口 smoke 输出：

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_eval_smoke_1profile/eval_summary.json
```

其中：

```text
has_prediction_npz = false
```

这是正常结果，因为当前还没有模型预测文件。

## D9 下一步建议

1. 新增条件化 effective SPM PINN 核心模块：

```text
gv1/model.py
gv1/output_transform.py
gv1/losses.py
gv1/trainer.py
scripts/gv1_train_conditioned_pinn.py
```

2. 先做 1 cell / short-window smoke，不要直接全量训练 24 个 profile。

3. 将 `T(t)` 作为 measured-temperature replay 输入保留；XJTU 文件中已有实测 temperature_C。

4. XJTU Batch-3/4 存在高倍率 / 高电流段，可能超出 effective SPM 假设，应分 protocol / C-rate / temperature 输出指标。

5. 后续所有模型选择仍必须遵守 no-test selection，不能用 test 指标选 checkpoint。

## 边界声明

- `ModelFin_112_deterministic_wrapper` 是 ASSB 当前五目标工程基线，不是跨电池泛化模型。
- GV1 D8 已完成数据通路和训练入口 scaffold，但尚未训练条件化 PINN。
- XJTU 负极材料在说明文件中未明确给出，当前 `graphite_assumed` 是工程假设。
- 高倍率段如果 effective SPM 误差明显，应考虑子集训练、adapter 或后续 P2D。
