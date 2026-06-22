# D18-S1 Array-level Diagnostic Schema

## 1. 输入要求

优先读取 D17-G6F/G7 已保存的 dense selected-cycle NPZ。一个文件至少需要：

```text
t_global_s / time_s
cycle_id
<state>_pred
<state>_true_report_only 或 <state>_true
```

可选但强烈建议：

```text
I_profile
voltage_exp
step_type
temperature_C
r_a / r_c
canonical_cell_uid
protocol
semantic_branch
split
```

支持状态：

```text
theta_a, theta_c, cs_a, cs_c, phie, phis_c
```

如果 prediction NPZ 不含 true arrays，脚本会尝试从：

1. NPZ 内 `source_softlabel_npz`；
2. D17 split manifest 中 `softlabel_npz`；

加载并按物理时间线性对齐。frozen test / test / flagged probe 默认禁止。

## 2. 输出表

### `d18_s1_state_metrics.csv`

每 case/state：

```text
MAE, RMSE, bias, exact R2, corr
NMAE/NRMSE by true range
std/range ratio
constant-shift corrected R2
best affine corrected R2
best integer-lag R2 gain
cycle-bias trend
residual SVD rank
```

### `d18_s1_radial_components.csv`

对 `cs/theta` 分解：

```text
volume mean / inventory
zero-mean radial deviation
surface-minus-center
radial amplitude
radial direction accuracy
zero-volume-mean audit
```

### `d18_s1_error_components_by_cycle.csv`

逐 cycle 输出 bias/MAE/RMSE，用于识别 early/middle/late 漂移与 history dependence。

### `d18_s1_cycle_boundary_audit.csv`

比较 true/pred 在相邻 cycle 边界的 jump，识别终态未正确传递的问题。

### `d18_s1_residual_rank.csv`

对 residual(time, radial) 做 SVD，报告 rank@90/95/99 与前 1/2/4/8 rank energy。

### `d18_s1_theta_cs_consistency.csv`

审计 theta 与 cs 是否仍保持同一 stoichiometry mapping；若 pred relation 显著劣于 teacher，不能进入训练。

## 3. 诊断标签规则

### `LOW_DIMENSIONAL_LATENT_SUFFICIENT`

失败状态中至少 75% 可以通过 constant/affine correction 恢复到指定 R²，且 residual rank@95 较低。

### `SEQUENCE_MODEL_REQUIRED`

任一成立：

- time-lag correction 显著提高 R²；
- cycle bias 与 cycle index 显著相关；
- residual rank@95 较高；
- cycle boundary jump 误差超门槛。

### `BRANCH_SPECIFIC_OPERATOR_REQUIRED`

RG/P4D failed-state R² 差距明显，或失败集中在 P4D/GEO branch。

### `TEACHER_OR_DATA_INCONSISTENCY`

出现 non-finite truth/pred，或 theta-cs relation 严重失配。

## 4. 防泄漏

默认允许：

```text
train
validation
internal_heldout
```

默认禁止：

```text
frozen_test
test
flagged_probe
unknown
```

若旧 NPZ 没有 split 字段，脚本会先用 D17 split manifest 和文件路径中的 canonical UID 解析；仍无法解析时保持 `unknown` 并拒绝选入，不能为方便而绕过 frozen-test 防线。

S1 用于 architecture diagnosis，不得用 frozen test 设计结构或超参数。

## 5. 没有自动找到 arrays 时

若 `status=REVIEW_NO_CASES`：

1. 打开 `d18_s1_case_inventory.csv`；
2. 检查 D17-G6F/G7 prediction 实际目录名；
3. 修改 `configs/d18_p0_s0_s1.json` 的 `s1.prediction_roots`；
4. 必要时设 `include_uids` 或 `cycle_ranges`；
5. 保持 `blocked_splits` 不变；
6. 重新运行 S1，不要因此直接启动 S2。
