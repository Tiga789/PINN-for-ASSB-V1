# D15-P1 · XJTU P2Dlite-RG 8-cell closed-set NN smoke

本包用于 D15-P1：在 **D15-P0 已通过的 P2Dlite-RG 8-cell soft labels** 上训练一个轻量 MLP，检查神经网络端是否可以复现增强后的径向状态标签。

## 边界

- 不修改 `ModelFin_112_deterministic_wrapper`。
- 不修改 D12-S1K voltage wrapper。
- 不覆盖旧 `xjtu_softlabels_p2dlite_v1_p4b_multicell_v3`、P5B、P5C。
- 本阶段是 **8-cell closed-set calibration / smoke**，不是 held-out 泛化。
- 输出仍是 P2Dlite-RG model-consistent soft labels 的 NN 复现，不是实验直接测得的内部状态真值。

## 默认输入

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_d15p0_8cell
```

该目录应由 D15-P0 生成，包含 8 个 `solution_softlabels.npz`。

## 默认输出

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p1_rg_closedset_nn_smoke
```

主要输出文件：

```text
D15_P1_PREFLIGHT.json
D15_P1_TRAINING_SUMMARY.json
D15_P1_DATASET_SAMPLING_SUMMARY.json
training_history.csv
model/best_with_state.pt
model/normalization_and_schema.npz
eval_full_profiles/D15_P1_EVAL_SUMMARY.json
eval_full_profiles/D15_P1_METRICS_BY_PROFILE.csv
D15_P1_FINAL_SCORECARD.json
```

## 一键运行

在项目根目录运行：

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
powershell -ExecutionPolicy Bypass -File scripts\d15_p1_run_all.ps1
```

如果需要重跑同一个 D15-P1 输出目录：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p1_run_all.ps1 -AllowOverwrite
```

快速 debug，不用于正式 scorecard：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p1_run_all.ps1 -AllowOverwrite -Quick
```

## 分步运行

```powershell
$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
$CacheRoot = "E:\XJTU battery dataset\_gv1_cache"
$SoftlabelDir = "$CacheRoot\xjtu_softlabels_p2dlite_rg_v1_d15p0_8cell"
$RunDir = "$CacheRoot\xjtu_d15_p1_rg_closedset_nn_smoke"
cd $ProjectRoot

python -m compileall -q gv1 scripts
python scripts\d15_p1_selftest_nn_smoke.py

python scripts\d15_p1_preflight.py `
  --softlabel-dir "$SoftlabelDir" `
  --prior-json "configs\P2Dlite_prior_xjtu_lr18650la_rg_v1.json" `
  --config "configs\d15_p1_nn_smoke_config.json" `
  --out-json "$RunDir\D15_P1_PREFLIGHT.json"

python scripts\d15_p1_train_rg_closedset_nn_smoke.py `
  --softlabel-dir "$SoftlabelDir" `
  --out-dir "$RunDir" `
  --config "configs\d15_p1_nn_smoke_config.json" `
  --allow-overwrite

python scripts\d15_p1_eval_rg_closedset_nn_smoke.py `
  --softlabel-dir "$SoftlabelDir" `
  --model-dir "$RunDir" `
  --out-dir "$RunDir\eval_full_profiles" `
  --config "configs\d15_p1_nn_smoke_config.json" `
  --allow-overwrite

python scripts\d15_p1_collect_scorecard.py `
  --run-dir "$RunDir" `
  --eval-dir "$RunDir\eval_full_profiles" `
  --out-json "$RunDir\D15_P1_FINAL_SCORECARD.json"
```

## 上传给我检查的结果

D15-P1 跑完后优先压缩以下轻量结果：

```powershell
$CacheRoot = "E:\XJTU battery dataset\_gv1_cache"
$RunDir = "$CacheRoot\xjtu_d15_p1_rg_closedset_nn_smoke"
$ReviewZip = "$CacheRoot\xjtu_d15_p1_results_for_review.zip"

Compress-Archive -Force `
  -Path `
    "$RunDir\D15_P1_PREFLIGHT.json", `
    "$RunDir\D15_P1_TRAINING_SUMMARY.json", `
    "$RunDir\D15_P1_DATASET_SAMPLING_SUMMARY.json", `
    "$RunDir\training_history.csv", `
    "$RunDir\eval_full_profiles\D15_P1_EVAL_SUMMARY.json", `
    "$RunDir\eval_full_profiles\D15_P1_METRICS_BY_PROFILE.csv", `
    "$RunDir\D15_P1_FINAL_SCORECARD.json" `
  -DestinationPath $ReviewZip
```

上传：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p1_results_for_review.zip
```

不需要上传 `model/best_with_state.pt`，除非我后面明确要求。

## PASS / REVIEW 解释

`D15_P1_FINAL_SCORECARD.json` 中：

- `final_status = PASS`：closed-set NN smoke 通过，可进入 D15-P2 精度 benchmark 或 RG 参数微调。
- `final_status = REVIEW`：脚本跑通，但至少一个 smoke 阈值未达标；应先看 `D15_P1_EVAL_SUMMARY.json` 的具体变量，不要直接加大训练。
- `final_status = FAIL`：缺关键文件或流程错误，需要先修脚本/路径。
