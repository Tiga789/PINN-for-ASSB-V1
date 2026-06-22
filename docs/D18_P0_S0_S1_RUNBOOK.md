# D18-P0/S0/S1 Windows Runbook

## A. 一次性覆盖后检查

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File .\VERIFY_D18_PACKAGE.ps1
```

预期：

```text
PASS: package manifest
PASS: compileall
PASS: unit tests
PASS: synthetic P0/S0/S1 dry-run
```

## B. 检查路径配置

```powershell
Get-Content .\configs\d18_p0_s0_s1.json
```

重点确认：

```text
project.project_root
paths.output_root
paths.d17_g_root
paths.d17_p_root
paths.all55_root
paths.d17_split_manifest
```

本包默认使用项目现有路径：

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
E:\XJTU battery dataset\_gv1_cache\xjtu_d17_g
E:\XJTU battery dataset\_gv1_cache\xjtu_d17_pinn_rebuild
E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL
E:\XJTU battery dataset\_gv1_cache\xjtu_d18_fullcycle
```

## C. 正式执行

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN_D18_P0_S0_S1.ps1
```

禁用绘图以加快诊断：

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN_D18_P0_S0_S1.ps1 -NoPlots
```

## D. 分阶段执行

### P0

```powershell
python .\scripts\d18_p0_freeze.py --config .\configs\d18_p0_s0_s1.json
```

### S0

```powershell
python .\scripts\d18_s0_validate_architecture.py --config .\configs\d18_p0_s0_s1.json
```

### S1

```powershell
python .\scripts\d18_s1_array_latent_diagnostic.py --config .\configs\d18_p0_s0_s1.json
```

## E. 结果读取顺序

1. `D18_P0_S0_S1_OVERALL_STATUS.md`
2. `d18_p0_freeze/P0_STATUS.md`
3. `d18_s0_architecture/S0_STATUS.md`
4. `d18_s1_array_diagnostic/d18_s1_recommendation.md`
5. `d18_s1_array_diagnostic/d18_s1_state_metrics.csv`
6. `d18_s1_array_diagnostic/d18_s1_radial_components.csv`
7. `d18_s1_array_diagnostic/d18_s1_cycle_boundary_audit.csv`

## F. Stop conditions

以下任一出现都必须停在 S1：

```text
P0 required artifact missing
S0 synthetic check FAIL
S1 no dense arrays selected
frozen_test_used=true
TEACHER_OR_DATA_INCONSISTENCY
known dense failure not reproduced
```

即使全部 PASS，也只代表可人工设计 S2，不代表可以直接长训练。

## G. 为最终 55-cell/all-cycle 目标保留的评估层

后续所有模型必须同时报告：

```text
1. D17-compatible sampled-grid metrics
2. dense selected-cycle metrics
3. cycle-wise full-profile streaming metrics
4. per-protocol and RG/P4D branch metrics
5. early/middle/late cycle metrics
6. charge/rest/discharge metrics
7. cycle-boundary state-continuity metrics
8. theta-cs consistency and radial zero-mean audit
```

最终 55-cell audit 采用 streaming metrics-only，避免保存 55-cell 全时序大 prediction 副本。
