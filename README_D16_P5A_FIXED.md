# D16-P5A fixed existing-model transfer evaluation package

## 为什么要用这个修正版

你遇到的实际问题是：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_D15P2_existing_on_ALL55\eval_full_profiles
```

为空，并且 `precision_audit` 里显示：

```text
prediction_file_count = 0
no prediction npz found under ...\eval_full_profiles\predictions
```

原因不是 ALL55 soft labels 坏了。你的 preflight 已经证明 ALL55 soft labels 是 55/55 且基础检查 PASS。

真正的问题是：原来的 D15-P2 评估链是给 8-cell closed-set benchmark 用的。D15-P2 模型带有 8-cell `profile_onehot` 输入。直接把 55 个 ALL55 profile 交给原 D15-P1/P2 evaluator 时，未见过的 profile 会被按 sorted order 赋予超出 one-hot 宽度的 index，导致 evaluator 在生成 prediction 之前失败；后续 audit 自然只能看到 0 个 prediction。

这个修正版做了三件事：

1. 先生成 ALL55 prediction，不再让 audit 对空目录运行。
2. 对 D15-P2 旧模型的 8-cell one-hot 输入做 **unseen profile routing**：exact match / same batch+protocol / same protocol / same batch / Batch-2 fallback / first-seen fallback。
3. 同时保存 raw 和 projected theta 结果；默认把 projected 版本写入 `eval_full_profiles\predictions`，让原 D15-P2 precision audit 能直接读取。

## 覆盖/新增文件

把本 zip 添加到项目根目录：

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
```

新增文件：

```text
scripts/gv1_d16_p5a_existing_transfer_eval_fixed.py
scripts/gv1_run_d16_p5a_fixed.ps1
README_D16_P5A_FIXED.md
PACKAGE_MANIFEST.json
```

不会覆盖 `gv1/`、`main.py`、`util/`、D9.6/D9.5.1、D12-S1K、D15_ALL55_FINAL 或已有模型目录。

## 推荐执行命令

### 1. 快速 smoke：先跑 2 个 cell，确认能生成 prediction

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5a_fixed.ps1 `
  -AllowOverwrite `
  -LimitCells 2 `
  -Device "cuda:0" `
  -BatchSize 65536
```

检查：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_FIXED_D15P2_existing_on_ALL55\eval_full_profiles\predictions
```

里面应当出现 `.npz` 文件。只要这里不为空，就说明之前的核心问题已经修复。

### 2. 正式 ALL55 evaluation

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5a_fixed.ps1 `
  -AllowOverwrite `
  -Device "cuda:0" `
  -BatchSize 65536
```

如果显存不足，把 `-BatchSize 65536` 改成 `32768` 或 `16384`。

## 默认路径

默认 soft-label 输入：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL
```

默认已有 D15-P2 模型目录：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p2_rg_precision_benchmark
```

默认 D16-P5A 输出：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_FIXED_D15P2_existing_on_ALL55
```

如果你的 D15-P2 模型目录不是这个路径，可以显式传入：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5a_fixed.ps1 `
  -ModelDir "E:\XJTU battery dataset\_gv1_cache\你的D15模型目录" `
  -AllowOverwrite
```

## 关键输出在哪里

### 总分卡

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_FIXED_D15P2_existing_on_ALL55\D16_P5A_FIXED_SCORECARD.json
```

重点字段：

```text
final_status
operational_status
metric_status_primary
raw_scorecard_status
projected_scorecard_status
precision_audit_status
profile_count_discovered
profile_count_predicted
```

解释：

```text
PASS   = 预测生成成功，并且指标/审计均过当前门槛
REVIEW = 预测生成成功，但旧模型迁移 ALL55 的某些指标需要诊断
FAIL   = 运行失败，例如没有生成 prediction 或文件不可读
```

D16-P5A 是 transfer evaluation。旧 8-cell 模型在 ALL55 上出现 REVIEW 并不奇怪；REVIEW 的意义是告诉你 D16-P5B 训练 ALL55 unified NN 的必要性。

### eval summary

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_FIXED_D15P2_existing_on_ALL55\eval_full_profiles\D16_P5A_EVAL_SUMMARY.json
```

这里有 raw/projected global metrics。

### prediction 输出

默认 primary 是 projected，所以原 D15-P2 audit 会读取这个目录：

```text
...\eval_full_profiles\predictions
```

另外保留：

```text
...\eval_full_profiles\predictions_raw
...\eval_full_profiles\predictions_projected
...\eval_full_profiles\predictions_raw_projected
```

### per-profile metrics

```text
...\eval_full_profiles\D16_P5A_METRICS_BY_PROFILE.csv
```

重点列：

```text
profile_id
projection_mode
batch
protocol
routed_profile_id
route_reason
phis_c_mae
phie_mae
theta_a_mae
theta_c_mae
grad_a_surface_center_mae
grad_c_surface_center_mae
pred_theta_outside_fraction
min_selected_corr
```

### routing table

```text
...\eval_full_profiles\D16_P5A_ROUTING_TABLE.csv
```

这个文件非常重要。它告诉你每个 ALL55 profile 被路由到 D15-P2 8-cell 模型中的哪个 seen profile one-hot。

重点列：

```text
profile_id
batch
protocol
routed_profile_index
routed_profile_id
route_reason
seen_exact
```

如果某个 batch/protocol 指标很差，先看它是否大量使用 fallback route。

### batch / protocol 汇总

```text
...\eval_full_profiles\D16_P5A_BATCH_METRICS.csv
...\eval_full_profiles\D16_P5A_PROTOCOL_METRICS.csv
```

用来判断问题是否集中在 Batch-2、Batch-5/6、GEO 或 random walk。

### precision audit

```text
...\precision_audit\D15_P2_PRECISION_AUDIT_SUMMARY.json
...\precision_audit\D15_P2_PRECISION_AUDIT_BY_PROFILE.csv
...\precision_audit\D15_P2_CYCLE_LEVEL_AUDIT.csv
...\precision_audit\D15_P2_TOPK_ERROR_WINDOWS.csv
```

这一步现在会有 prediction 文件可读，不再是 0 profile。

## 结果怎么判断

先看：

```text
D16_P5A_FIXED_SCORECARD.json
```

- `operational_status = PASS` 且 `profile_count_predicted = 55`：说明 D16-P5A 执行链跑通。
- `projected_scorecard_status` 比 `raw_scorecard_status` 好：说明 projection 对 ALL55 有帮助。
- `final_status = REVIEW`：通常表示旧 D15-P2 模型迁移 ALL55 不够稳，下一步应进入 D16-P5B。
- `final_status = FAIL`：先不要分析物理指标，先看 `failures` 和 `eval_full_profiles\D16_P5A_FAILURES.csv`。

再看：

```text
D16_P5A_ROUTING_TABLE.csv
```

如果 Batch-2 或 Batch-5/6 指标差，先判断 route 是否属于 fallback。fallback 越多，越说明旧 8-cell 模型不具备 ALL55 泛化能力。

最后看：

```text
D16_P5A_BATCH_METRICS.csv
D16_P5A_PROTOCOL_METRICS.csv
```

这两个文件用于决定 D16-P5B 是训练一个 ALL55 unified NN，还是 batch/protocol-aware expert。
