# D17-P3.4 final forward-core promotion patch

This patch is intentionally small. It assumes the D17-P3.3 package has already been installed, and it only adds/overwrites files required for P3.4.

## What P3.4 changes

P3.3 proved that the forward-core reliability audit pipeline runs, but it could not be promoted because the forward electrochemical core still relied on a placeholder prior and the residual budget remained too large. P3.4 addresses that by:

1. Building an aligned D17-compatible resolved P2Dlite-RG spec from prior JSON candidates plus observed replay `I(t), V(t), T(t)`.
2. Never reading `cs/theta/phie/phis` state soft-label arrays during spec alignment or training.
3. Centering profile latent choices at the aligned prior: `theta_a0`, `theta_c0`, `Q_eff`, `R_ohm`, and `bV`.
4. Adding a `forward_voltage` loss so the electrochemical core itself must explain terminal voltage.
5. Keeping D12-S1K transition-fade as a bounded residual, not as an unbounded voltage copier.

## Files in this patch

```text
gv1/d17_pinn/p2dlite_prior.py
gv1/d17_pinn/latent_adapter.py
gv1/d17_pinn/model.py
gv1/d17_pinn/losses.py
gv1/d17_pinn/p34_resolved_spec.py
scripts/d17_p34_final_forward_core_promotion.py
scripts/d17_p34_inspect_summary.py
configs/d17_pinn_rebuild_p34_final_forward_core_promotion.json
configs/d17_pinn_rebuild_p34_final_forward_core_promotion.yaml
docs/D17_P34_FILE_LIST_ACTUAL.txt
README_D17_P34_FINAL_FORWARD_CORE_PROMOTION.md
```

## Run

```powershell
python scripts\d17_p34_final_forward_core_promotion.py `
  --config configs/d17_pinn_rebuild_p34_final_forward_core_promotion.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --base_resolved_spec "configs/resolved_p2dlite_spec_placeholder.json" `
  --softlabel_root "E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL" `
  --replay_search_root "E:/XJTU battery dataset/_gv1_cache" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p34_final_forward_core_promotion" `
  --profile_count 12 `
  --validation_profile_count 6 `
  --max_time_points 512 `
  --time_window_s 40000 `
  --n_r 17 `
  --epochs 220 `
  --warmup_epochs 30 `
  --voltage_recovery_until_epoch 155 `
  --validation_adaptation_steps 120 `
  --lr 0.0009 `
  --device auto
```

Inspect:

```powershell
python scripts\d17_p34_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p34_final_forward_core_promotion/D17_P34_FINAL_FORWARD_CORE_PROMOTION_SUMMARY.json"
```

## Promotion rule

Continue to P4 only if:

```text
status = PASS
promotion_status = PASS
p4_ready = true
validation_forward_voltage_mae_mean_V <= 0.09
validation_corrected_voltage_mae_mean_V <= 0.06
residual_budget_status = PASS
training_uses_state_softlabels = false
validation_adaptation_uses_state_softlabels = false
```

If `status=PASS` but `promotion_status=REVIEW`, do not enter P4. The summary will list `p4_blockers`.
