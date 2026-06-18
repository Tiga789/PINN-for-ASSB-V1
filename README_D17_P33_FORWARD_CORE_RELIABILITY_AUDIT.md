# D17-P3.3 Forward-Core Reliability Audit

本包是 D17-P3.3 完整覆盖包，目标是把 D12-S1K 的 low/transition-fade 电压经验以公式形式迁移到 D17，同时把 `V_forward` 与 residual-corrected `V_pred` 分开审计。

## 这轮解决什么

P3.2 已经把 corrected voltage 压低，但 residual 太强，不能说明 forward electrochemical core 本体可靠。P3.3 因此新增：

1. D12-S1K-style low/transition fade formula：`gv1/d17_pinn/d12_transition_fade.py`。
2. `V_forward`、`V_residual_total`、`V_pred` 三路输出审计。
3. 12 train + 6 validation voltage-only check。
4. validation observed-only profile adaptation：冻结 model weights，只用 validation 的 I/V/T 优化 profile latent 和少量 smooth basis coefficient，不使用 state soft labels。
5. residual budget audit：corrected voltage 通过不等于 promotion；若 residual 太大，`promotion_status=REVIEW`。

## 覆盖后运行

```powershell
python scripts\d17_p33_forward_core_reliability_audit.py `
  --config configs/d17_pinn_rebuild_p33_forward_core_reliability.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --resolved_spec "configs/resolved_p2dlite_spec_placeholder.json" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p33_forward_core_reliability_audit" `
  --profile_count 12 `
  --validation_profile_count 6 `
  --max_time_points 512 `
  --time_window_s 40000 `
  --n_r 17 `
  --epochs 170 `
  --warmup_epochs 25 `
  --voltage_recovery_until_epoch 125 `
  --validation_adaptation_steps 90 `
  --lr 0.0009 `
  --device auto
```

## 检查 summary

```powershell
python scripts\d17_p33_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p33_forward_core_reliability_audit/D17_P33_FORWARD_CORE_RELIABILITY_SUMMARY.json"
```

## 主要输出

```text
D17_P33_FORWARD_CORE_RELIABILITY_SUMMARY.json
D17_P33_FORMULA_ALIGNMENT_AUDIT.json
D17_P33_RESIDUAL_BUDGET_AUDIT.json
training_history_train.csv
validation_adaptation_history.csv
selected_profiles_train.json
selected_profiles_validation.json
model/best_model_and_latents.pt
predictions/D17_P33_TRAIN_PROFILE_*.npz
predictions/D17_P33_VALIDATION_PROFILE_*.npz
```

## 判读规则

- `status=PASS`：P3.3 审计流程、no-state-label 边界、zero-mean 和 bounds 没触发阻断。
- `promotion_status=PASS`：forward core 和 residual budget 均达到目标；这比 `status=PASS` 更严格。
- `promotion_status=REVIEW`：不是代码失败，而是说明 corrected voltage 仍可能主要由 residual 解释。

## No-state-label 边界

训练与 validation adaptation 只读 replay NPZ 中的 observed fields：`t_global_s/I_profile/voltage_exp/temperature_C/...`。Manifest 中的 `softlabel_npz` 只作为 report-only 路径保留，不由 P3.3 trainer 加载。
