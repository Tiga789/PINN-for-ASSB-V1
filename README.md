# PINN-for-ASSB-V1 / QJW-2

本项目基于 PINNSTRIPES / effective SPM 思路，先完成 NMC811||Li-In 全固态电池（ASSB）的五目标工程基线，再推进 GV1 / XJTU 数据集的 measured-current replay 泛化能力。当前最新工作阶段为 **ASSB-D10 / GV1 D10-D12 metadata ablation 复盘阶段**。

## 当前总状态

### ASSB 本体基线

当前 ASSB 五目标工程统一基线仍为：

```text
ModelFin_112_deterministic_wrapper
EvalFin_112_deterministic_wrapper
```

该基线来自 D7：

- 四个电化学状态 `cs_a / cs_c / phie / phis_c` 来自 frozen `ModelFin_107A` state eval NPZ。
- SOH 来自 `ModelFin_112_deterministicSOH_ridge_g4`。
- 这是一个 engineering wrapper / unified package，不是端到端联合训练的单个神经网络。
- 它证明当前 ASSB 本电池五目标工程封装成立，但不代表跨电池规格泛化。

### GV1 / XJTU 当前主线

GV1 当前稳定主线是：

```text
D9.6 / D9.5.1 trend-first warmup rare-regime conditioned effective-SPM PINN
```

D10 的主要结论是：

```text
D9.6/D9.5.1 accepted for non-outlier 23-profile 200ks verification.
B1_2C battery-8 remains flagged/excluded as late-2C discharge boundary/outlier.
No D10-P3 correction is adopted.
```

关键数值：

| item | result |
|---|---:|
| D10-P1 profile_count | 23 |
| pass / borderline / fail / read_error | 23 / 0 / 0 / 0 |
| mean_MAE_V | 0.0658758364 |
| mean_RMSE_V | 0.0983696769 |
| mean_corr | 0.9521213381 |

## Battery-8 状态

`B1_2C battery-8` 当前不是普通失败 profile，而是已登记的 flagged regime/outlier case：

```text
battery8_status = flagged_excluded_late_2C_discharge_boundary_outlier
```

证据概要：

- D10-P0 判定：`battery8_flagged_late_2C_discharge_regime_outlier_keep_D9_6_mainline`。
- charge 段较好：MAE≈0.038 V，corr≈0.967。
- discharge 段明显差：MAE≈0.148 V，corr≈0.720。
- B1_2C peer outlier：battery-8 的 40ks MAE 相对 peers MAD-z≈6.13。
- D10-P3 没有找到安全 lightweight correction，最终保持 `identity_d9_6_raw`。
- D11-B feature distance audit 支持 boundary/regime 解释：abs_z_ge_5_count=117，abs_z_ge_3_count=181。

## D11 / D12 metadata ablation 状态

D11-C / D11C2 完成了 flag-aware metadata 的 design-only 与 metadata input patch contract，但不改变主线。

D12 完成 metadata off / zero / on runtime ablation：

- `metadata_off`: D9.6/D9.5.1 reference，不追加 metadata。
- `metadata_zero`: 追加同维度零 metadata，作为 architecture-control。
- `metadata_on`: 追加 D11C2/D12 enriched metadata values。

### D12-S1 3-profile strict smoke

| mode | n | ok | mean_MAE_V | mean_corr | mean_bias_V |
|---|---:|---:|---:|---:|---:|
| off | 3 | 3 | 0.0738773362 | 0.9353194522 | -0.0212812764 |
| on | 3 | 3 | 0.0688431833 | 0.9339160011 | 0.0221299927 |
| zero | 3 | 3 | 0.0903647797 | 0.9299126442 | -0.0593920881 |

Interpretation: `metadata_on` showed a small positive MAE signal on 3 2C profiles, but this was only a short-window smoke result.

### D12-S2 balanced 6-profile strict smoke

D12-S2 selected 2 profiles each from 2C / R2.5 / R3, excluding battery-8.

| mode | n | ok | mean_MAE_V | mean_RMSE_V | mean_corr | mean_bias_V |
|---|---:|---:|---:|---:|---:|---:|
| off | 6 | 6 | 0.0754177362 | 0.1083145151 | 0.9348272578 | -0.0239817897 |
| on | 6 | 6 | 0.0901967342 | 0.1239603726 | 0.9324948258 | -0.0342953316 |
| zero | 6 | 6 | 0.0763487702 | 0.1134116999 | 0.9314697878 | -0.0146021228 |

Interpretation:

```text
D12 metadata runtime path is validated.
metadata_on is NOT better than metadata_off on balanced 6-profile smoke.
metadata_on should NOT replace the D9.6/D9.5.1 mainline.
```

## 当前可以说明什么

可以说明：

```text
XJTU Batch-1/3/4 的代表性 2C/R2.5/R3 profiles 在 40ks short window 上可以被 GV1 主线稳定拟合和评估。
D12 runtime metadata on/off/zero ablation pipeline 已经跑通。
```

不能说明：

```text
XJTU 三个 batch 全量数据电压拟合已经完成。
24-profile 或 23-profile metadata_on 泛化验证已经完成。
metadata_on 比主线更好。
```

## 当前不要做

```text
Do not run direct 24-profile 200ks mainline.
Do not unflag battery-8.
Do not overwrite D9.6/D9.5.1.
Do not adopt D9.6.1 / D9.6.2 / D9.6.3 hard/component guard repairs.
Do not promote D12 metadata_on to mainline based on D12-S1.
Do not reuse old D12 runtime generated scripts with 40000 epochs / 200ks.
Do not treat D12-S3 as completed; it has not been executed in this window.
```

## 关键目录

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_train_conditioned_pinn_23x200ks_d10p1_exclude_battery8
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_d10_p0_battery8_regime_judgement
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_d10_p3_battery8_lightweight_correction
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_d10_p5_regime_policy_d11_plan
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_d11_b_regime_feature_distance_audit
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_d11c2_metadata_input_patch_design
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_d12_metadata_on_off_ablation_plan
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_d12_s1_metadata_ablation_scorecard
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_d12_s2_metadata_ablation_scorecard
```

## D11 新窗口建议

1. 先阅读 `ASSB-D10_项目进度复盘总结_重新整理版.docx`。
2. 冻结并备份 D10/D12 结果，尤其是 D10-P1 与 D12-S2 scorecard。
3. 如果继续 D12-S3，请重新生成 clean 23-profile 40ks ablation 包，必须 strict preflight：

```text
epochs = 100
time_window_s = 40000
max_time_points = 1024
batch_size = 512
```

4. D12-S3 只能作为 ablation audit，不是 mainline replacement。
5. 若目标是证明 XJTU 三 batch 电压拟合可用，优先复核 D9.6/D10 主线 23-profile 40ks/200ks，而不是直接扩大 metadata_on。
6. metadata_on 当前不应替代 D9.6/D9.5.1；若继续研究 metadata，应分析 D12-S2 segment metrics 中高倍率 profile 退步的原因。

## 项目复盘文档索引

```text
ASSB-D1.docx  ASSB 先验与早期训练失败
ASSB-D2.docx  cycle5_v4 / ModelFin_101 闭环
ASSB-D3.docx  all-cycle soft labels 与 ModelFin_102/103
ASSB-D4.docx  v2 massclosed / ModelFin_106/107A
ASSB-D5.docx  SOH StageB 与 hybrid benchmark
ASSB-D6.docx  ModelFin_111 strict30 SOH
ASSB-D7.docx  ModelFin_112 deterministic wrapper
ASSB-D8.docx  GV1/XJTU data pipeline
ASSB-D9.docx  GV1 conditioned PINN / D9.6 / D9.7
ASSB-D10_项目进度复盘总结_重新整理版.docx  D10-P0 至 D12-S2 metadata ablation
```
