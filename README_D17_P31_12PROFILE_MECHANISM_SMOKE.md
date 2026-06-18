# D17-P3.1 12-profile mechanism smoke

This package is a complete D17-P3.1 overlay.  It includes the currently required D17-P3 fix2 mechanism code plus the new P3.1 12-profile smoke trainer, so it should be applied as one package rather than stacked with earlier partial patches.

## Purpose

D17-P3.1 extends the passed P3 fix2 mechanism smoke to 12 balanced train profiles and adds a voltage-recovery / loss-scale audit phase.  It keeps the D17 boundary:

- training source is `replay_npz` observed-only profile data;
- `cs/theta/phie/phis` soft-label arrays are not loaded as inputs or training losses;
- `softlabel_npz` paths remain report-only metadata;
- `Batch-1_2C_battery-8` / flagged_probe records are refused if selected;
- `cs = cbar + zero-volume-mean delta_c` is preserved by hard inventory projection and bounded radial residual scaling.

## Main command

```powershell
python scripts\d17_p31_mechanism_smoke_12profile.py `
  --config configs/d17_pinn_rebuild_p31_12profile_smoke.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --resolved_spec "configs/resolved_p2dlite_spec_placeholder.json" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p31_12profile_mechanism_smoke" `
  --profile_count 12 `
  --max_time_points 512 `
  --time_window_s 40000 `
  --n_r 17 `
  --epochs 120 `
  --warmup_epochs 20 `
  --voltage_recovery_until_epoch 80 `
  --lr 0.0007 `
  --device auto
```

## Inspect command

```powershell
python scripts\d17_p31_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p31_12profile_mechanism_smoke/D17_P31_12PROFILE_SMOKE_SUMMARY.json"
```

## Key outputs

```text
D17_P31_12PROFILE_SMOKE_SUMMARY.json
D17_P31_LOSS_SCALE_AUDIT.json
training_history.csv
selected_profiles.json
model/best_model_and_latents.pt
model/last_model_and_latents.pt
predictions/D17_P31_PROFILE_*.npz
```

## Status interpretation

`status=PASS` means the 12-profile smoke ran, no state labels were used, battery-8 was not selected, zero-mean / theta-bounds audits passed, and voltage MAE is below the configured review threshold.  `voltage_target_met=true` is stricter: it means the mean voltage MAE reached the current P3.1 target of 0.09 V.

If status is `REVIEW`, inspect:

```text
voltage_recovery.final_voltage_mae_mean_V
final_aggregate.zero_mean_max_abs_*_mol_m3_max
final_aggregate.theta_*_min/max
D17_P31_LOSS_SCALE_AUDIT.json
```

Do not proceed to full train until mechanism audits remain clean and the voltage loss scale is understood.
