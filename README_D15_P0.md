# D15-P0 · XJTU P2Dlite radial-gradient audit + P2Dlite-RG generator

本包用于 QJW-2 / PINN-for-ASSB-V1 的 D15-P0：先审计 D14 P4B-v3 P2Dlite v1 soft labels 的径向梯度，再生成新的 P2Dlite-RG（Radial-Gradient-aware）soft labels。包内文件只新增，不修改旧 ASSB 主线、D9.6/D12-S1K 电压主线、P4B/P5B/P5C 输出。

## 1. 默认路径

旧输入目录（只读）：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_v1_p4b_multicell_v3
```

新输出目录（D15-P0 新建，不覆盖旧目录）：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_d15p0_8cell
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p0_radial_gradient_audit_p2dlite_v1
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p0_radial_gradient_audit_p2dlite_rg_v1
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p0_radial_gradient_rg_comparison
```

## 2. 覆盖位置

把本包解压后的内容复制到项目根目录：

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
```

新增文件位置：

```text
configs/P2Dlite_prior_xjtu_lr18650la_rg_v1.json
scripts/d15_p0_preflight.py
scripts/d15_p0_radial_gradient_audit.py
scripts/d15_p0_generate_p2dlite_rg_softlabels.py
scripts/d15_p0_compare_radial_audits.py
scripts/d15_p0_selftest_rg_solver.py
scripts/d15_p0_run_all.ps1
gv1/p2dlite_rg/__init__.py
gv1/p2dlite_rg/io_utils.py
gv1/p2dlite_rg/radial_solver.py
gv1/p2dlite_rg/audit.py
```

## 3. 一键运行

在 PowerShell 中运行：

```powershell
cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
powershell -ExecutionPolicy Bypass -File scripts\d15_p0_run_all.ps1
```

如果你明确要重跑并覆盖 D15-P0 自己的输出目录，可以加：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p0_run_all.ps1 -AllowOverwrite
```

不建议用 `-AllowOverwrite` 指向旧 P4B/P5B/P5C 目录；preflight 会阻止明显的旧目录覆盖。

## 4. 运行后看哪些文件

旧标签审计：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p0_radial_gradient_audit_p2dlite_v1\radial_gradient_audit_summary.json
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p0_radial_gradient_audit_p2dlite_v1\radial_gradient_audit_by_profile.csv
```

新 P2Dlite-RG 标签：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_d15p0_8cell\D15_P0_RG_GENERATION_REPORT.json
每个 profile 子目录\solution_softlabels.npz
每个 profile 子目录\soft_label_summary.json
```

新标签审计与对比：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p0_radial_gradient_audit_p2dlite_rg_v1\radial_gradient_audit_summary.json
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p0_radial_gradient_rg_comparison\D15_P0_RADIAL_OLD_VS_RG_COMPARISON_SUMMARY.json
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p0_radial_gradient_rg_comparison\D15_P0_RADIAL_OLD_VS_RG_COMPARISON.csv
```

## 5. P2Dlite-RG 做了什么

P2Dlite-RG 把固相浓度写成：

```text
c_s(t,r) = cbar(t) + delta_c(t,r)
```

其中：

- `cbar(t)` 仍来自原 P2Dlite v1 的平均库存，保证容量/质量守恒不被破坏；
- `delta_c(t,r)` 由球形有限体积隐式扩散步、表面通量、固相扩散系数和零体积均值投影生成；
- `surface-center` 梯度方向按 I>0 充电约定审计：正极应为 `-sign(I)`，负极应为 `+sign(I)`；
- `phis_c`、`phie`、D12-S1K 电压一致性标签在 D15-P0 中默认保留，不在这一步重调电压；
- 新输出仍是 **model-consistent soft labels**，不是实验直接测得的内部状态真值。

## 6. D15-P0 通过标准

D15-P0 不是最终训练阶段，建议通过标准为：

```text
1. preflight overall_status = PASS
2. old audit 能明确显示 P2Dlite v1 径向梯度过弱（WARN/FAIL 可接受）
3. RG generation overall_status = PASS
4. RG audit read_error_count = 0，且无 FAIL；若有 WARN，先看对应 profile 的 direction/mass_cbar 指标
5. comparison 中 RG active_abs_gradient_norm_p95 明显高于 old，同时 mass_cbar_mae_norm 不显著变坏
```

通过后再进入 D15-P1：用 `xjtu_softlabels_p2dlite_rg_v1_d15p0_8cell` 做 8-cell closed-set NN smoke / precision benchmark，验证 NN 能否复现增强后的 `cs_a/cs_c` 径向结构。
