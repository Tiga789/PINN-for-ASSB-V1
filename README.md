# PINN-for-ASSB-V1 / QJW-2

本项目基于 PINNSTRIPES / effective SPM / P2Dlite 思路，先完成 NMC811||Li-In 全固态电池（ASSB）的内部状态与 SOH 工程基线，再推进 XJTU / Kokam / 多路径泛化。当前最新工作阶段为 **ASSB-D16 / XJTU P2Dlite-RG ALL55 strict protocol reset 前夜**。

## 当前总状态

### ASSB 基线

当前 ASSB 五目标工程统一基线仍为：

```text
ModelFin_112_deterministic_wrapper
EvalFin_112_deterministic_wrapper
```

该基线来自 D7：

- 四个电化学状态 `cs_a / cs_c / phie / phis_c` 来自 frozen `ModelFin_107A` state eval NPZ。
- SOH 来自 deterministic ridge SOH。
- 这是一个 engineering wrapper / unified package，不是端到端联合训练的单个神经网络。
- 它不代表跨电池规格泛化。

### XJTU soft-label 基线

D15 已完成 XJTU 55-cell P2Dlite-RG model-consistent soft-label 全覆盖：

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL
```

D15 summary：

```text
expected_cell_count = 55
actual_cell_count   = 55
missing_count       = 0
extra_count         = 0
status              = PASS
total_size_GB       ≈ 52.742
```

重要边界：P2Dlite-RG labels 是 model-consistent soft labels，不是实验直接测得的内部状态真值。不能在论文或 README 中写成 “55 个 cell 的真实 cs_a/cs_c 径向状态”。

### XJTU D16 NN / PINN 状态

D16 完成了从 P5B 到 P5K-G 的一系列 train/eval/audit。最终结论如下：

```text
P5K-G formal training:
  operational_status      = PASS
  profile_count_evaluated = 55
  failure_count           = 0
```

但是 P5K-G **不能 promotion**。

P5K-G 的 normal eval43 有明显改善：

```text
eval theta_a_mean_mae = 0.113033
eval theta_a_mean_r2  = 0.640233
eval theta_c_mean_mae = 0.102038
eval theta_c_mean_r2  = 0.594277
eval phis_c_r2        = 0.999516
```

但 hard_probe 仍不合格：

```text
hard_probe theta_a_mean_mae = 0.101127
hard_probe theta_a_mean_r2  = -0.128193
hard_probe theta_c_mean_mae = 0.107799
hard_probe theta_c_mean_r2  = -0.556818
```

core_train / hard-like profiles 也仍失败：

```text
core_train theta_a_mean_r2 = -1.000656
core_train theta_c_mean_r2 = -1.400700
```

因此当前定位是：

```text
P5K-G = operational PASS + normal eval improvement + hard_probe/core_train failure
P5K-G is a development artifact, not a final four-state model.
```

## D16 的关键原则性结论

### 1. 8-cell closed-set 高精度不是泛化证据

D14/D15 的 8-cell P5B/P5C precision 是 closed-set calibration benchmark：8 个 profile 全部参与训练和评估。它只能证明 NN 架构和数据管线可以复现已见 profile 的 P2Dlite/P2Dlite-RG labels，不能证明 held-out cell generalization。

若训练和评估使用同一批 soft-label state arrays，则从泛化评估角度等同于模型已经见过答案。

### 2. soft-label generator 只能作验证工具

真实目标不是训练 NN 去复现 soft-label generator，而是训练一个在真实场景中只依赖：

```text
I(t), V(t), t, T, protocol, cell spec, OCP/capacity/geometry priors, physics residuals
```

的 physics-constrained neural model，直接输出：

```text
cs_a, cs_c, phie, phis_c
```

soft-label generator 只能作为：

```text
offline validation / reference solution / credibility audit
```

不能作为：

```text
training label source
oracle theta0 provider
rule/gate tuning feedback on frozen test
```

### 3. 当前路线存在 validation-guided design leakage

虽然 P5K-G 训练本身没有使用 `theta/cs/phie/phis_c` state data loss，但 G0-G4 过程中反复使用 ALL55 soft-label feedback 设计 theta0 gate/rule。这不等于 training-loss 数据泄露，但属于 validation-guided design leakage。D17 必须停止这种做法。

### 4. exact R² 是硬指标

P5H exact-R² 审计证明：P5B/P5D/P5E/P5F/P5G 虽然 `phis_c` 很好，但 theta absolute R² 均为负。corr/corr_mean 不能作为内部状态成功依据。后续 promotion 必须看 exact MAE/R²。

### 5. P5K-G 的训练破坏了 G4 hard baseline

G4 no-training baseline：

```text
G4-rule_v2_strict_aggressive hard_probe:
  theta_a_mean_mae = 0.074450
  theta_a_mean_r2  = 0.218311
  theta_c_mean_mae = 0.075991
  theta_c_mean_r2  = 0.013939
```

P5K-G 正式训练后 hard_probe 退化为负 R²。这说明 residual NN 没有受到 hard-profile no-regression / no-drift 约束，不能继续直接加 epoch。

## 当前不应继续做什么

不要继续：

```text
- P5K-G 追加 epoch
- 在 ALL55 soft labels 上继续设计 G5/G6/G7 rule/gate
- 用 8-cell closed-set precision 证明泛化
- 用 soft-label oracle shift 设计正式 no-state-label 模型
- 把 P5K-G 写成 final high-precision four-state model
```

## D17 推荐工作路线

D17 应进行 **strict no-state-label protocol reset**。

### D17-P0：冻结 D16 状态

保留并归档：

```text
D16_P5H_EXACT_R2_AUDIT_REPORT.md
D16_P5KG0_BASELINE_REPAIR_AUDIT_REPORT.md
D16_P5KG1_OBSERVED_THETA0_AUDIT_REPORT.md
D16_P5KG2_GATED_THETA0_ADAPTER_AUDIT_REPORT.md
D16_P5KG3_THETA0_ADAPTER_V2_AUDIT_REPORT.md
D16_P5KG4_EXACT_ARRAY_AUDIT_REPORT.md
D16_P5KG_FINAL_SCORECARD.json
D16_P5KG_SPLIT_METRICS.csv
D16_P5KG_METRICS_BY_PROFILE.csv
```

P5K-G 只作为 development artifact，不 promotion。

### D17-P1：固定 train / validation / frozen test

后续规则：

```text
- development/train set 可用于训练和调参
- validation set 可用于 checkpoint selection
- frozen test set 只允许最终一次性评估
- 禁止根据 frozen test soft-label 结果修改 rule/gate/loss
```

### D17-P2：严格 no-state-label PINN

训练允许使用：

```text
t_global_s
I_profile
voltage_exp
temperature_C
protocol / batch metadata
cell spec
OCP prior
capacity / geometry prior
physics residuals
```

训练和规则设计禁止使用：

```text
theta_a / theta_c
cs_a / cs_c
phie / phis_c / phis_c_soft
oracle_shift
theta0_oracle
frozen-test soft-label feedback
```

### D17-P3：模型结构原则

后续 NN 不应直接自由输出完整浓度曲面。推荐结构：

```text
c_s,j(t,r) = cbar_j(t) + δc_j(t,r)
```

其中：

```text
cbar_j(t) 由 I(t) 积分和质量守恒硬编码
δc_j(t,r) 由 NN 输出，但必须 zero-volume-mean
```

必须加入：

```text
- diffusion PDE residual
- center symmetry residual
- surface flux boundary residual
- zero-volume-mean radial residual
- capacity/current conservation
- voltage closure
- initial-condition enforcement
```

### D17-P4：如果使用 soft-label state data loss，必须改名

如果后续允许使用少量 `theta/cs/phie/phis_c` soft labels 来训练 theta0 adapter 或 residual NN，则实验必须明确写成：

```text
supervised / semi-supervised soft-label calibration
```

不能继续称为 no-state-label PINN。

## 当前关键路径

```text
Project root:
C:/Users/Tiga_QJW/Desktop/ASSB_Scheme_V1/PINN-for-ASSB-V1

XJTU cache root:
E:/XJTU battery dataset/_gv1_cache

D15 ALL55 soft labels:
E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL

P5K-G formal run:
E:/XJTU battery dataset/_gv1_cache/xjtu_d16_p5kg_rulev2_strict_gate_FAST/G_train12_rulev2_strict
```

不要删除：

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL/profiles
```

可以清理的通常是：

```text
_p5k*_mmap_cache
失败实验 eval_all55_vs_softlabels 大目录
旧失败 run 的 prediction dumps
```

清理前必须保留 scorecard、split metrics、by-profile metrics、training summary 和 input audit。

## 新窗口接续提示

新窗口开始时请先阅读：

```text
ASSB-D16_项目进度复盘总结_20260614.docx
README.md
ASSB-D15.docx
ASSB-D14.docx
D16_P5KG4_EXACT_ARRAY_AUDIT_REPORT.md
D16_P5KG_FINAL_SCORECARD.json / SPLIT_METRICS.csv
```

当前最重要任务：

```text
不要继续 P5K-G 加 epoch。
不要继续在 ALL55 soft labels 上调 rule。
先设计 D17 strict no-state-label train/val/frozen-test protocol。
```
