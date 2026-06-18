# D17-G6F selected-cycle on-demand inference + plotting

This package adds a lightweight workflow for selected-cycle inference, metrics and 3D plotting.

It does **not** train, does **not** select checkpoints, and does **not** run a 55-cell full audit.

## Intended use

Use a frozen D17-G candidate to predict a requested cell/cycle range, compute report-only metrics against the released ALL55 soft labels, and optionally plot interactive 3D surfaces for `cs_a` and `cs_c`.

The default workflow keeps prediction arrays in memory. It saves only metrics and figures unless `--save_temp_npz` or `--keep_temp_npz` is explicitly set.

## Typical command

```powershell
python scripts\d17_g6f_selected_cycle_infer_plot.py `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --candidate_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g21_p4d_branch_repair" `
  --candidate_summary "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g21_p4d_branch_repair/D17_G21_P4D_BRANCH_REPAIR_SUMMARY.json" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g6f_selected_cycle_infer_plot/Batch2_battery3_cycle13_15" `
  --batch 2 `
  --battery 3 `
  --cycles 13-15 `
  --metric_targets cs_a cs_c phie phis_c `
  --plot_targets both `
  --plot_3d `
  --save_png `
  --device auto
```

If Qt is available and the window is not interactive, add:

```powershell
--backend QtAgg
```

or use:

```powershell
--backend TkAgg
```

## Inspect summary

```powershell
python scripts\d17_g6f_inspect_selected_cycle_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g6f_selected_cycle_infer_plot/Batch2_battery3_cycle13_15/D17_G6F_SELECTED_CYCLE_SUMMARY.json"
```

## Decision field

The summary includes:

```text
full_training_recommendation
```

- `NO_FULL_TRAINING_INDICATED_FOR_THIS_SELECTED_RANGE`: the requested cycle range meets the configured local gates.
- `SELECTED_RANGE_BELOW_GATE_REVIEW_BEFORE_FULL_TRAINING`: the requested range fails local gates. Review the metrics before deciding whether a broader full-cycle training/audit is warranted.

This is a selected-range decision only. It is not a proof of all-cell/all-cycle success.
