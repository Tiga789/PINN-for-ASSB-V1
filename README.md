# PINN-for-ASSB-V1 / QJW-2

本项目基于 PINNSTRIPES / effective SPM 思路，先完成 NMC811||Li-In 全固态电池（ASSB）的五目标工程基线，再推进 XJTU/Lishen 18650 NCM523 数据集上的泛化能力。当前最新工作阶段为 **ASSB-D12 / GV1 low-anchor voltage wrapper 与 P2D 参数先验阶段**。

## 当前总状态

### ASSB 本体基线

当前 ASSB 五目标工程统一基线仍为：

```text
ModelFin_112_deterministic_wrapper
EvalFin_112_deterministic_wrapper
```

该基线来自 D7：四个电化学状态 `cs_a / cs_c / phie / phis_c` 来自 frozen `ModelFin_107A` state eval NPZ，SOH 来自 `ModelFin_112_deterministicSOH_ridge_g4`。它是 engineering wrapper / unified package，不是端到端联合训练的单个神经网络，也不是跨电池规格泛化证明。

### GV1 / XJTU 当前电压基线

D12 当前最推荐的 GV1 电压结果为：

```text
D12-S1K two-candidate wrapper
主候选：d12s1k_low_plus_transition_fade_to_baseline
保守对照：d12s1k_low_only_revert_nonlow_to_baseline
```

该结果建立在 D9.6 / D9.5.1 trend-first warmup rare-regime conditioned effective-SPM-style voltage surrogate 之上，使用 S1E low-anchor correction 的低压能力，并通过 low-only / transition-fade wrapper 限制 correction 向 normal/rest/high 区域泄漏。

当前结论必须严格写成：

```text
D12-S1K 在排除 / flag Batch-1_2C_battery-8 后的 23-profile 200ks voltage replay / voltage surrogate 上通过。
```

不能写成：

```text
24/24 所有电池全部成功。
cs_a / cs_c / phie / phis_c 已经是真实内部状态标签。
已经完成完整 P2D 参数辨识。
```

## D12 最重要结果

### D12-S1K 23-profile 200ks confirmation

范围：XJTU Batch-1/3/4，排除 `Batch-1_2C_battery-8`，共 23 个 non-outlier profiles。

```text
prediction_count = 69
metrics_ok_count = 69
read_error_count = 0
promoted_candidates = 2
```

两个通过候选：

```text
d12s1k_low_only_revert_nonlow_to_baseline
d12s1k_low_plus_transition_fade_to_baseline
```

关键均值指标：

| mode | mean_MAE_V | mean_corr | 说明 |
|---|---:|---:|---|
| baseline_d951 | 0.057825 | 0.960949 | D9.6/D9.5.1 baseline |
| low_only_revert_nonlow_to_baseline | 0.055519 | 0.970704 | 最保守、解释性最强 |
| low_plus_transition_fade_to_baseline | 0.054621 | 0.973269 | 当前主推荐 |

`transition_fade` 的关键增益：

```text
global MAE 改善约 -3.204 mV
low_target 改善约 -333.9 mV
low<=2.75 改善约 -431.9 mV
normal / high 不退化
rest 改善约 -7.67 mV
corr 提升约 +0.01232
```

## D12 解决的问题

D12 解决了以下问题：

1. 确认 D11 的 P2D-like / low-anchor 方向不是错的，低压修正本身有效。
2. 确认 S1E 失败根因不是 low-anchor 失效，而是 correction 泄漏到 high / normal / rest / non-low 区。
3. 通过 S1H/S1I/S1J diagnostic-only wrappers 快速定位 high leakage 与 200ks normal/rest leakage。
4. 通过 S1K low-only / transition-fade wrapper，在 23-profile 40ks 和 23-profile 200ks 上完成 confirmation。
5. 建立了电压 surrogate 与内部状态 soft labels 的边界：当前仅证明电压工程反演，不证明真实 P2D 内部状态。
6. 对 Batch-1_2C_battery-8 做了数据诊断，确认它不是明显坏文件，但仍是 flagged outlier / special regime。
7. 根据 XJTU Data Introduction 与公开资料，整理了 P2D 参数先验表和可调整范围，为 D13 内部状态/P2D 路线做准备。

## 尚未解决的问题

### 1. Batch-1_2C_battery-8 仍未解决

当前异常 profile 是：

```text
Batch-1_2C_battery-8
```

它属于 Batch-1 固定 2C 充电 / 1C 放电策略组。D12 battery-8 data diagnosis 显示：

```text
time_nonmonotonic_count = 0
nan_count_time/current/voltage/temperature = 0
large_dt_gap_count_gt_60s = 0
voltage_outside_2p4_4p7_count = 0
verdict = undetermined
```

但它相对同批 peers 明显偏离：

```text
charge_duration_s robust_z ≈ 3.45
n_time_points robust_z ≈ 2.87
duration_s robust_z ≈ 2.87
energy_discharge_Wh robust_z ≈ 2.19
q_discharge_Ah robust_z ≈ 2.08
cycle_count robust_z ≈ 2.02
```

因此当前判断：它不像明显数据坏文件，更像 cell-specific behavior / special regime / model boundary。D13 不应把它强行纳入 23-profile 主线。

### 2. 内部状态软标签尚不可靠

XJTU 说明文件只明确给出：

```text
manufacturer = Lishen
format = 18650
positive chemistry = LiNi0.5Co0.2Mn0.3O2 / NCM523
nominal capacity = 2 Ah
nominal voltage = 3.6 V
voltage window = 4.2 V / 2.5 V
sampling = 1 Hz
Batch-1/3/4 protocol
```

但没有给出完整 P2D 所需参数：

```text
positive/negative/separator thickness
porosity
tortuosity / Bruggeman
particle radius
electrode area
OCP curves
solid/electrolyte diffusivity
ionic conductivity
reaction rate / exchange current density
initial stoichiometry window
```

因此 D12 当前成果不能证明 `cs_a / cs_c / phie / phis_c` 是真实物理状态。后续只能先称为 model-consistent latent states，并应通过 P2D prior + grouped calibration + identifiability analysis 建立置信等级。

## 关键目录

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_d12_s1k_two_candidate_23x200ks_scorecard
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_d12_s1e_p2d_anchor_budget_23x200ks
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_d12_battery8_data_diagnosis
```

D12-S1K 复查重点文件：

```text
D12_S1K_scorecard_summary.json
D12_S1K_candidate_decisions.csv
D12_S1K_mode_summary.csv
D12_S1K_segment_metrics.csv
D12_S1K_run_metrics.csv
D12_S1K_source_leakage_overview.csv
D12_S1K_RECOMMENDATION.md
```

battery-8 诊断重点文件：

```text
D12_B8_diagnostic_summary.json
D12_B8_RECOMMENDATION.md
D12_B8_profile_peer_summary.csv
D12_B8_robust_peer_outlier_scores.csv
D12_B8_target_anomaly_events.csv
D12_B8_target_segment_summary.csv
D12_B8_target_cycle_summary.csv
```

## D13 推荐路线

D13 不建议继续在 S1K low wrapper 上反复调参。优先级如下：

1. 冻结 D12-S1K 结果，作为当前非 battery-8 voltage surrogate baseline。
2. 继续 flag `Batch-1_2C_battery-8`，不要纳入 23-profile 主线。
3. 如果要解释 battery-8，先审计 `voltage_jump_gt_150mV` 附近是否都在 step transition。
4. 如果要做内部状态预测，先建立 XJTU NCM523/graphite P2D 参数 prior YAML。
5. 做 grouped calibration / identifiability，不要直接硬拟合几十个 P2D 参数。
6. 设计 shared temporal encoder + chemistry/model-specific expert paths，而不是用同一个物理输出头套所有电池。
7. 可选：对 S1K transition-fade 做 23-profile 500ks audit，但这不是 D13 的必要前置。

## 当前边界声明

- `ModelFin_112_deterministic_wrapper` 是 ASSB 当前五目标工程基线，不是跨电池泛化模型。
- `D12-S1K` 是 XJTU non-outlier 23-profile voltage replay / voltage surrogate 的当前推荐 wrapper。
- `Batch-1_2C_battery-8` 仍然 flagged，不属于当前主线成功范围。
- 当前还没有可靠的 XJTU P2D 内部状态软标签。
- 后续若生成 `cs_a / cs_c / phie / phis_c`，必须写明参数来源、假设等级和 label confidence。
