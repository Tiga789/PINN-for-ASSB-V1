# PINN-for-ASSB-V1 / QJW-2

本项目基于 PINNSTRIPES / effective SPM 思路，先完成 NMC811||Li-In 全固态电池（ASSB）的五目标工程基线，再推进不同电池与不同充放电策略下的泛化能力。当前最新工作阶段为 **ASSB-D9 / GV1 条件化 effective SPM PINN 与 multi-profile verification 阶段**。

## 当前总状态

### ASSB 本电池五目标基线

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

### D8 / GV1 数据管线

D8 已经完成 XJTU Batch-1/3/4 measured-current replay 数据管线：

| 项目 | 结果 |
|---|---:|
| 原始 `.mat` 文件数 | 24 |
| replay profile NPZ | 24 |
| cycle 行数 | 13513 |
| 完整放电 SOH 标签数 | 8787 |
| partial/unlabeled cycle | 4726 |
| profile split | train 18 / val 3 / test 3 |
| training-ready linked rows | 13513 / 13513 |

### D9 / GV1 条件化 PINN 当前状态

D9 已经完成条件化 effective SPM PINN 的核心训练层：

```text
gv1/model.py
gv1/output_transform.py
gv1/profile_adaptive.py
gv1/losses.py
gv1/trainer.py
scripts/gv1_train_conditioned_pinn.py
```

当前 GV1 主线应保留为：

```text
D9.5.1 trend-first warmup rare-regime core
+
D9.6 multi-cell / multi-profile verification package
```

不要使用 D9.6.1、D9.6.2、D9.6.3 覆盖主线；这些版本均已在 battery-8 200ks 诊断中失败。

## D9 关键结果

### D9.5.1 三 profile 500ks

| run | MAE (V) | RMSE (V) | corr | bias (V) | 结论 |
|---|---:|---:|---:|---:|---|
| B1_2C_500ks | 0.0315 | 0.0533 | 0.9866 | +0.0026 | 很好，低压覆盖基本对齐 |
| B3_R25_500ks | 0.0701 | 0.1058 | 0.9515 | -0.0092 | 通过，低压尾段仍弱 |
| B4_R3_500ks | 0.0586 | 0.0832 | 0.9651 | -0.0074 | 通过 |

D9.5.1 是 D9 阶段最稳定的单/三 profile 主线。

### D9.6 multi-profile verification

| 验证 | profile 数 | pass / borderline / fail | mean MAE (V) | mean RMSE (V) | mean corr | mean bias (V) | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| 6-profile 40ks | 6 | 6 / 0 / 0 | 0.0680 | 0.1038 | 0.9379 | -0.0195 | 通过 |
| 6-profile 200ks | 6 | 6 / 0 / 0 | 0.0603 | 0.0918 | 0.9579 | -0.0088 | 通过 |
| 24-profile 40ks | 24 | 23 / 1 / 0 | 0.0713 | 0.1070 | 0.9345 | -0.0166 | borderline_continue_carefully |

唯一 borderline profile 是：

```text
B1_2C_battery-8_0008_battery-8_2C_battery-8_40ks
```

### Battery-8 200ks outlier 诊断

原始 D9.6 battery-8 200ks：

```text
MAE  = 0.1008 V
RMSE = 0.1482 V
corr = 0.8926
bias = +0.0300 V
voltage_pred_max ≈ 4.476 V
pred_upper_frac_ge_4p269 ≈ 0.0104
pred_overshoot_frac_gt_4p35 ≈ 0.0043
```

D9.7 分段诊断显示：

```text
charge_I_pos:     MAE≈0.0380 V, corr≈0.967
 discharge_I_neg: MAE≈0.1479 V, corr≈0.720
```

因此 battery-8 的主要问题集中在放电段 / mid-target / high-target / high-voltage overshoot，不是所有电流段都失败。

## 已拒绝的 D9 分支

| 分支 | 失败原因 | 处理 |
|---|---|---|
| D9.6.1 targeted late-2C repair | 高压大量饱和，`pred_upper_frac_ge_4p269≈0.36–0.39` | 不使用 |
| D9.6.2 rollback-style repair | 电压动态范围塌缩到 `3.796–3.856 V` | 不使用 |
| D9.6.3 training-strategy repair | 降低学习率 / 换 seed 未改善，3 个候选均 fail | 不使用 |
| D9.6.4 safe-strategy adjustment | 本窗口没有形成已验证真实可用交付和测试记录 | 不使用 |

## 当前推荐路线

当前不要直接运行：

```text
24-profile 200ks
24-profile 500ks
任何 D9.6.1 / D9.6.2 / D9.6.3 替代主线训练
```

D10 应从 D9.7 诊断继续：

1. 保留 D9.6 / D9.5.1 主线。
2. 阅读 `d97_battery8_diagnosis_summary.json`、`d97_plot_manifest.json`、`d97_candidate_metrics_table.csv`。
3. 判断 battery-8 是否属于 outlier / late-2C regime / effective SPM 表达边界。
4. 再决定是否：
   - 标记 battery-8 为 outlier；
   - 做 23-profile 200ks excluding battery-8；
   - 做 battery-8 profile-specific post-hoc calibration；
   - 或设计 D9.8 discharge-regime specific 轻量修正。

## 关键命令

### 查看 6-profile 200ks scorecard

```powershell
Get-Content "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_multicell_6x200ks_d96\scorecard_d96_200ks.json" -Raw
```

### 查看 24-profile 40ks scorecard

```powershell
Get-Content "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_multicell_24x40ks_d96\scorecard_d96_40ks.json" -Raw
```

### 查看 battery-8 原始 D9.6 200ks 指标

```powershell
Get-Content "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_borderline_B1_2C_battery8_200ks_d96\metrics_borderline_200ks.json" -Raw
```

### 查看 D9.7 diagnosis summary

```powershell
Get-Content "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d97_battery8_outlier_diagnosis\d97_battery8_diagnosis_summary.json" -Raw
```

### 查看 D9.7 candidate CSV

```powershell
Import-Csv "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d97_battery8_outlier_diagnosis\diagnosis_plots\d97_candidate_metrics_table.csv" |
  Format-Table label,mae_V,rmse_V,bias_V,corr,pred_upper_frac_ge_4p269,pred_overshoot_frac_gt_4p35 -AutoSize
```

## 工程规则

- 用户会手动解压压缩包并添加/覆盖到项目中；后续说明只给覆盖后检查、运行和读取命令。
- 不要修改旧 ASSB 主线文件 `main.py`、`util/*`、`integration_spm/*`。
- GV1 代码继续保持独立路径：`gv1/`、`scripts/gv1_*`。
- 大文件缓存继续放在：

```text
E:/XJTU battery dataset/_gv1_cache
```

## 给 D10 的一句话状态

ASSB-D9 已经完成 GV1 条件化 effective SPM PINN 从核心训练层到 multi-profile verification 的闭环；D9.5.1/D9.6 是当前主线，6x40ks 与 6x200ks 全 pass，24x40ks 为 23 pass + 1 borderline。唯一阻塞项是 B1_2C battery-8，在 200ks 下出现放电段趋势不足和少量高压 overshoot。D9.6.1/6.2/6.3 均失败，D9.7 已完成 outlier/regime 诊断。D10 应先判断 battery-8 的数据/工况边界，不要直接运行 24-profile 200ks。
