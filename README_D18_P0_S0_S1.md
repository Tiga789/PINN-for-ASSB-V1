# D18-P0_S0_S1：55 cells × full cycles 前置工程包

版本：`D18-P0_S0_S1-v1.0.0`  
目标项目：`QJW-2 / PINN-for-ASSB-V1`  
最终目标：**XJTU 55 cells × all cycles 的高精度 full-cycle / arbitrary-cycle generator surrogate**

## 1. 本包做什么

本包只完成 D18 的三个前置阶段，不进行模型训练：

- **D18-P0**：冻结 D17-G4/G5、G6F、G7-S0/S1/S1E、D15-P4D exact-replay 脚本/config/replay manifest、ALL55 manifest/summary 等证据，记录路径、大小、SHA256 和本地 Git 状态。
- **D18-S0**：验证新的 cycle-aware / full-profile operator 架构及其 hard output transform；只运行 synthetic forward，不优化权重。
- **D18-S1**：读取现有 dense selected-cycle prediction/true arrays，分解 inventory、radial、cycle drift、time lag、boundary jump、phie gauge 和 RG/P4D branch failure。

本包明确不会：

- 重新生成 55-cell soft labels；
- 修改或覆盖 `gv1/d17_g`；
- 继续 G7-S1 checkpoint；
- 启动 D18-S2 训练；
- 使用 frozen test 参与架构选择；
- 把 P2Dlite-RG soft labels 表述为实验内部状态真值。

## 2. 覆盖位置

将压缩包内容直接放到项目根目录：

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
```

本包只新增：

```text
configs/d18_p0_s0_s1.json
gv1/d18_cycleaware/*
scripts/d18_*.py
docs/D18_*.md
RUN_D18_P0_S0_S1.ps1
VERIFY_D18_PACKAGE.ps1
README_D18_P0_S0_S1.md
PACKAGE_MANIFEST.json
PACKAGE_FILELIST.txt
```

## 2.1 默认防泄漏规则

- S1 只读取 split manifest 中明确标记为 `train`、`validation` 或 `internal_heldout` 的数组。
- `frozen_test`、`test`、`flagged_probe` 和 `unknown` 默认全部阻断。
- 本包不固定本地 Git SHA；P0 如实记录本地 HEAD、分支与工作区状态，避免因本地提交领先 GitHub 而误报失败。

## 3. 覆盖后先验证

在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\VERIFY_D18_PACKAGE.ps1
```

该命令执行：

1. `PACKAGE_MANIFEST.json` 文件 SHA256 校验；
2. Python `compileall`；
3. 核心单元测试；
4. P0→S0→S1 synthetic end-to-end dry-run。

只有出现 `PASS` 后再运行正式 P0/S0/S1。

## 4. 正式运行

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN_D18_P0_S0_S1.ps1
```

默认输出：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d18_fullcycle
```

自定义输出目录：

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN_D18_P0_S0_S1.ps1 `
  -OutputRoot "E:\XJTU battery dataset\_gv1_cache\xjtu_d18_fullcycle_run1"
```

只运行 Python 主入口也可以：

```powershell
python .\scripts\d18_run_p0_s0_s1.py `
  --config .\configs\d18_p0_s0_s1.json
```

## 5. 关键输出

```text
xjtu_d18_fullcycle/
  d18_p0_freeze/
    p0_freeze_manifest.json
    p0_git_state.json
    p0_artifact_status.csv
    P0_STATUS.md

  d18_s0_architecture/
    d18_s0_architecture_contract.json
    d18_s0_validation.json
    S0_STATUS.md

  d18_s1_array_diagnostic/
    d18_s1_case_inventory.csv
    d18_s1_state_metrics.csv
    d18_s1_phase_metrics.csv
    d18_s1_error_components_by_cycle.csv
    d18_s1_cycle_boundary_audit.csv
    d18_s1_residual_rank.csv
    d18_s1_radial_components.csv
    d18_s1_theta_cs_consistency.csv
    d18_s1_array_latent_summary.json
    d18_s1_recommendation.md
    plots/

  D18_P0_S0_S1_OVERALL_SUMMARY.json
  D18_P0_S0_S1_OVERALL_STATUS.md
```

## 6. S1 诊断标签解释

- `LOW_DIMENSIONAL_LATENT_SUFFICIENT`：常数/仿射 latent 加少量低秩 residual 可能足够。
- `SEQUENCE_MODEL_REQUIRED`：需要 cycle-history / within-cycle sequence encoder，不能继续逐点 MLP。
- `BRANCH_SPECIFIC_OPERATOR_REQUIRED`：RG 与 P4D/GEO 必须保留独立 adapter/head。
- `TEACHER_OR_DATA_INCONSISTENCY`：先检查 teacher/provenance/array alignment，不能训练。
- `STRUCTURAL_OPERATOR_REDESIGN_REQUIRED`：失败不属于简单 shift、lag 或低秩补偿，需要 operator 架构重构。

即使 S1 为 PASS，`go_to_s2` 仍固定为 `false`；必须人工审查报告后再单独交付 S2。

## 7. 面向 55 cells × all cycles 的后续门槛

本包完成后，后续应按以下顺序推进：

1. **D18-S2**：4–8 train profiles + 2 validation，100–200 epochs，cycle-stratified smoke；
2. **D18-S3**：TCN/GRU/DeepONet-like operator 对比；
3. **D18-S4**：39 train / 7 validation mini-expansion，frozen test 不参与选择；
4. **D18-S5**：55-cell/all-cycle streaming metrics-only audit；
5. 同时通过 sampled-grid、dense selected-cycle、cycle-wise streaming 三套门槛后，才能写 all-cycle claim。

## 8. 当前架构的核心约束

- cycle-level full-profile GRU：学习 early/middle/late aging/history；
- causal within-cycle GRU：学习 charge/rest/discharge 局部状态；
- deterministic `cbar` baseline：平均库存不交给 MLP；
- zero-volume-mean radial basis：径向 residual 不修改库存；
- `theta` 从 `cs` 推导，不设独立冲突 head；
- shared potential gauge + differential heads：避免 phie/phis_c 共模漂移；
- RG/P4D branch-specific adapter：不再强制两类 generator semantics 共用同一 residual map。

更详细说明见：

```text
docs/D18_S0_ARCHITECTURE.md
docs/D18_S1_DIAGNOSTIC_SCHEMA.md
docs/D18_P0_S0_S1_RUNBOOK.md
```
