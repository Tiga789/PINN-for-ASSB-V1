# D15-P2 · XJTU P2Dlite-RG 8-cell closed-set precision benchmark

本包用于在已通过的 D15-P1 smoke 基础上，执行更严格的 P2Dlite-RG 8-cell closed-set NN precision benchmark。

## 定位

D15-P2 只检验神经网络是否能高精度复现 D15-P0 生成的 P2Dlite-RG soft labels。它不是 held-out 泛化证明，也不是实验测得的真实径向内部状态证明。

## 默认输入

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_d15p0_8cell
```

## 默认输出

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p2_rg_precision_benchmark
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p2_results_for_review.zip
```

## 依赖

D15-P2 复用已通过的 D15-P1 NN 基础模块，因此本地需要已有：

```text
gv1\p2dlite_rg_nn\*.py
scripts\d15_p1_train_rg_closedset_nn_smoke.py
scripts\d15_p1_eval_rg_closedset_nn_smoke.py
```

如果你已经成功运行过 D15-P1，则这些文件已经存在。

## 一键运行

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
powershell -ExecutionPolicy Bypass -File scripts\d15_p2_run_all.ps1
```

重跑：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p2_run_all.ps1 -AllowOverwrite
```

快速 debug：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p2_run_all.ps1 -AllowOverwrite -Quick
```

正式结果不要用 `-Quick`。

## 分步运行

```powershell
$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
$CacheRoot = "E:\XJTU battery dataset\_gv1_cache"
$SoftlabelDir = "$CacheRoot\xjtu_softlabels_p2dlite_rg_v1_d15p0_8cell"
$RunDir = "$CacheRoot\xjtu_d15_p2_rg_precision_benchmark"
$EvalDir = "$RunDir\eval_full_profiles"
$AuditDir = "$RunDir\precision_audit"
cd $ProjectRoot

python scripts\d15_p2_selftest_precision_benchmark.py

python scripts\d15_p2_preflight.py `
  --softlabel-dir "$SoftlabelDir" `
  --prior-json "configs\P2Dlite_prior_xjtu_lr18650la_rg_v1.json" `
  --config "configs\d15_p2_precision_benchmark_config.json" `
  --out-json "$RunDir\D15_P2_PREFLIGHT.json"

python scripts\d15_p2_train_rg_precision_benchmark.py `
  --softlabel-dir "$SoftlabelDir" `
  --out-dir "$RunDir" `
  --config "configs\d15_p2_precision_benchmark_config.json" `
  --allow-overwrite

python scripts\d15_p2_eval_rg_precision_benchmark.py `
  --softlabel-dir "$SoftlabelDir" `
  --model-dir "$RunDir" `
  --out-dir "$EvalDir" `
  --config "configs\d15_p2_precision_benchmark_config.json" `
  --allow-overwrite

python scripts\d15_p2_precision_audit.py `
  --softlabel-dir "$SoftlabelDir" `
  --eval-dir "$EvalDir" `
  --out-dir "$AuditDir" `
  --config "configs\d15_p2_precision_benchmark_config.json" `
  --allow-overwrite

python scripts\d15_p2_collect_scorecard.py `
  --run-dir "$RunDir" `
  --eval-dir "$EvalDir" `
  --audit-dir "$AuditDir" `
  --out-json "$RunDir\D15_P2_FINAL_SCORECARD.json"
```

## 结果检查

重点看：

```text
D15_P2_FINAL_SCORECARD.json
D15_P2_EVAL_SUMMARY.json
D15_P2_PRECISION_AUDIT_SUMMARY.json
D15_P2_PRECISION_AUDIT_BY_PROFILE.csv
D15_P2_TOPK_ERROR_WINDOWS.csv
D15_P2_CYCLE_LEVEL_AUDIT.csv
```

状态解释：

```text
PASS   = 可作为 D15-P2 precision benchmark 通过
REVIEW = 跑通但存在 review 级指标，需要人工判断
FAIL   = 流程或硬阈值失败，不进入下一阶段
```

## 不要覆盖的基线

D15-P2 不应修改或覆盖：

```text
ModelFin_112_deterministic_wrapper
D12-S1K voltage wrapper
xjtu_softlabels_p2dlite_v1_p4b_multicell_v3
P5B-v2 / P5C benchmark
xjtu_softlabels_p2dlite_rg_v1_d15p0_8cell
xjtu_d15_p1_rg_closedset_nn_smoke
```
