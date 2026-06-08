
# D14-P2: XJTU Generalization Scorecard Package

## 定位

D14-P2 只做 **scorecard 汇总与审计**，不训练模型、不改 `gv1/model.py`、不改 `gv1/output_transform.py`、不生成 SOH 标签。

它读取 D14-P0 / D14-P1 输出，以及既有 D10-P1 / D12-S1K scorecard 目录，生成统一的：

```text
global scorecard
by-protocol scorecard
by-cell scorecard
by-segment scorecard
candidate comparison
outlier policy
```

## 关键边界

- XJTU 当前支撑的是 non-outlier measured-current voltage replay / voltage surrogate generalization。
- D12-S1K `low_plus_transition_fade_to_baseline` 是当前推荐 voltage-wrapper candidate。
- `Batch-1 / 2C / battery-8` 继续 flagged/excluded，只作为 stress-test。
- `Batch-3_battery-8` 与 `Batch-4_battery-8` 不能因为名字包含 battery-8 就被误排除。
- XJTU voltage soft-label generator 不生成 SOH；SOH 从 XJTU 原始 cycle/capacity 数据读取或计算。
- XJTU voltage replay 成功不等于内部状态真值验证。

## 运行命令

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File .\scripts\run_gv1_d14_p2_generalization_scorecard.ps1 `
  -ProjectRoot "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  -CacheRoot "E:\XJTU battery dataset\_gv1_cache" `
  -P0Dir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p0_freeze_audit_v2" `
  -P1Dir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p1_evidence_boundary_v2" `
  -OutputDir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p2_generalization_scorecard" `
  -AllowWarn
```

如果你希望缺少 D10/D12 原始 scorecard 目录时直接失败，可以增加：

```powershell
  -StrictEvidence
```

## 预期输出

```text
D14_P2_GENERALIZATION_SCORECARD_REPORT.json
D14_P2_GENERALIZATION_SCORECARD_REPORT.md
D14_P2_SOURCE_INVENTORY.csv
D14_P2_RUN_METRICS_NORMALIZED.csv
D14_P2_SEGMENT_METRICS_NORMALIZED.csv
D14_P2_GLOBAL_SCORECARD.csv
D14_P2_BY_PROTOCOL.csv
D14_P2_BY_CELL.csv
D14_P2_BY_SEGMENT.csv
D14_P2_BY_PROTOCOL_SEGMENT.csv
D14_P2_CANDIDATE_COMPARISON.csv
D14_P2_OUTLIER_POLICY.csv
D14_P2_OUTPUT_INDEX.json
D14_P2_RUN_SUMMARY.txt
```

运行后请把输出目录压缩上传检查。
