# D17-G2 held-out generator-surrogate expansion

## 定位

D17-G2 是在 D17-G1.5R 通过后的扩展实验。它不再是小样本诊断，也不进入 frozen-test；它使用 D17 split 中全部 normal train profiles 的 soft labels 训练 generator surrogate，并用 validation profiles 做 report-only 审计。

## 前置条件

必须先满足：

```text
D17-G1.5R status = PASS
g2_ready = true
recommendation = ENTER_D17_G2_HELDOUT_SURROGATE_EXPANSION
```

## 策略

```text
train-cell soft labels: 用于训练 loss
validation soft labels: 只 report-only，不参与训练/选 checkpoint
frozen-test soft labels: 不读取
checkpoint selection: fit-train + protocol/branch-stratified train-internal heldout
profile conditioning: observed profile encoder，不使用 train profile-id 记忆
```

G2 默认：

```text
train_profile_count = 39
validation_profile_count = 7
internal_heldout_count = 8
max_time_points = 512
time_window_s = 40000
```

## 运行命令

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

python scripts\d17_g2_heldout_surrogate_expansion.py `
  --config configs/d17_g2_heldout_surrogate_expansion.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g2_heldout_surrogate_expansion" `
  --train_profile_count 39 `
  --validation_profile_count 7 `
  --internal_heldout_count 8 `
  --max_time_points 512 `
  --time_window_s 40000 `
  --epochs 1200 `
  --lr 0.0005 `
  --batch_size 1024 `
  --device auto
```

检查：

```powershell
python scripts\d17_g2_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g2_heldout_surrogate_expansion/D17_G2_HELDOUT_SURROGATE_EXPANSION_SUMMARY.json"
```

## 判读

```text
status = PASS
```

表示 G2 流程和 fit-train 基础复现通过。

```text
g3_ready = true
```

才表示可以进入 D17-G3 frozen-test report-only audit。

如果：

```text
status = PASS
g3_ready = false
```

不要进入 G3，把 summary 发回做 targeted repair。

## 主要输出

```text
D17_G2_HELDOUT_SURROGATE_EXPANSION_SUMMARY.json
D17_G2_STRATIFIED_SPLIT_AUDIT.csv
D17_G2_PROFILE_METRICS.csv
D17_G2_PER_TARGET_PROFILE_METRICS.csv
D17_G2_PER_TARGET_AGGREGATE.csv
D17_G2_PHIE_AUDIT.csv
D17_G2_PROFILE_ENCODER_FEATURE_AUDIT.csv
D17_G2_TARGET_NORMALIZATION_AUDIT.csv
D17_G2_PREDICTION_MANIFEST.csv
D17_G2_training_history.csv
model/best_model.pt
```
