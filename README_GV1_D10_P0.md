# GV1 D10-P0：battery-8 outlier / regime 判定包

本包是 **诊断优先** 的下一步，不修改 `gv1/model.py`、`gv1/output_transform.py`、`gv1/losses.py`、`gv1/trainer.py`，也不改旧 ASSB 主线 `main.py / util/* / integration_spm/*`。

## 包含文件

```text
scripts/gv1_d10_p0_battery8_regime_judgement.py
scripts/gv1_d10_metrics_from_prediction.py
scripts/gv1_d10_p1_prepare_23profile_200ks_plan.py
scripts/gv1_d10_p1_collect_scorecard.py
scripts/run_gv1_d10_p0_battery8_judgement.ps1
scripts/run_gv1_d10_p1_prepare_23profile_plan.ps1
manifests/d10_p0_expected_outputs.json
README_GV1_D10_P0.md
```

## D10-P0 要回答的问题

D9 已经确认：

- D9.5.1 / D9.6 是当前主线。
- D9.6.1、D9.6.2、D9.6.3 都失败，不应继续覆盖主线。
- 24-profile 40ks 为 23 pass + 1 borderline，唯一 borderline 是 B1_2C battery-8。
- battery-8 200ks 原始 D9.6 不是完全崩溃，但放电段明显弱，并伴随少量高压 overshoot。

D10-P0 的目标是把 battery-8 判定为以下之一：

```text
1. 数据/工况 outlier
2. late-2C 特殊放电 regime
3. effective SPM 表达边界
4. 证据不足，需要补充 peer/segment 诊断
```

## 操作步骤

以下命令均在项目根目录执行：

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
```

### 1. 确认 D9.6 主线不要被失败分支覆盖

```powershell
Select-String -Path "scripts\gv1_train_conditioned_pinn.py" -Pattern "D9.5.1|trend-first|warmup|rare-regime"
Select-String -Path "gv1\trainer.py" -Pattern "D9.5.1|trend-first|warmup"
Select-String -Path "gv1\output_transform.py" -Pattern "enable_voltage_hard_clamp|voltage_range_strategy|softsign"
```

预期：训练入口仍能看到 D9.5.1 / trend-first warmup 字样；不要出现你刚覆盖过的 D9.6.1 / D9.6.2 强 guard 主线逻辑。

### 2. 运行 D10-P0 判定脚本

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d10_p0_battery8_judgement.ps1"
```

如果希望同时生成 PNG 图：

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d10_p0_battery8_judgement.ps1" -MakePlots
```

### 3. 查看输出

默认输出目录：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d10_p0_battery8_regime_judgement
```

重点查看：

```powershell
Get-Content "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d10_p0_battery8_regime_judgement\D10_P0_RECOMMENDATION.md" -Raw
Get-Content "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d10_p0_battery8_regime_judgement\d10_p0_battery8_judgement_summary.json" -Raw
Import-Csv "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d10_p0_battery8_regime_judgement\d10_p0_candidate_metrics_normalized.csv" | Format-Table label,mae_V,rmse_V,corr,bias_V,pred_max_V,pred_upper_frac_ge_4p269,pred_overshoot_frac_gt_4p35,candidate_class -AutoSize
Import-Csv "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d10_p0_battery8_regime_judgement\d10_p0_segment_metrics_table.csv" | Format-Table label,mae_V,rmse_V,corr,bias_V -AutoSize
```

### 4. 判断通过标准

若 `verdict` 为：

```text
battery8_flagged_late_2C_discharge_regime_outlier_keep_D9_6_mainline
```

下一步进入 **D10-P1：23-profile 200ks excluding/flagging battery-8**。

若 `verdict` 为：

```text
battery8_regime_specific_discharge_issue_peer_evidence_incomplete
```

先补充 peer 对比，不要跑 24-profile 200ks。

若 `verdict` 为：

```text
inconclusive_need_more_battery8_peer_and_segment_evidence
```

先人工查看 D9.7 plots，不要继续训练修模型。

## 可选 D10-P1：只生成 23-profile 200ks 计划，不立即训练

当 D10-P0 结论确认 battery-8 应 flag/exclude 后，生成 23-profile 训练计划：

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d10_p1_prepare_23profile_plan.ps1"
```

该命令只生成计划和一个 `.generated.ps1`，不会自动训练。

默认计划目录：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d10_p1_23profile_200ks_plan
```

查看生成计划：

```powershell
Get-Content "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d10_p1_23profile_200ks_plan\d10_p1_prepare_23profile_plan_summary.json" -Raw
Import-Csv "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d10_p1_23profile_200ks_plan\d10_p1_23profile_manifest_excluding_battery8.csv" | Select-Object -First 5 | Format-Table
```

真正运行 23-profile 200ks 时，再执行生成的：

```powershell
powershell -ExecutionPolicy Bypass -File "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d10_p1_23profile_200ks_plan\run_d10_p1_23profile_200ks_excluding_battery8.generated.ps1"
```

## 当前不建议做的事

```text
不要直接运行 24-profile 200ks。
不要用 D9.6.1 / D9.6.2 / D9.6.3 覆盖 D9.6 主线。
不要继续加强 hard guard / component clamp。
不要在 battery-8 性质未判定前设计 D9.8 大改结构。
```
