# D17-G interactive 3D concentration surface plotter

This package adds one read-only plotting script:

```text
scripts/d17_g_plot_cs3d_interactive.py
```

It opens Matplotlib 3D popup windows that can be dragged/rotated with the toolbar/mouse.  For each selected concentration target, it plots two surfaces side by side:

```text
left  = PINN / D17-G surrogate prediction, default cmap=coolwarm
right = P2Dlite-RG soft-label truth, default cmap=viridis
```

The title contains global metrics over the selected cycles and all radial points:

```text
R², NMAE, NRMSE, MAE, RMSE, bias
```

The script is read-only. It does not train, overwrite, or modify model artifacts.

## Typical command

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

python scripts\d17_g_plot_cs3d_interactive.py `
  --batch 2 `
  --battery 3 `
  --cycles 13-15 `
  --target both `
  --pred_root "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g" `
  --softlabel_root "E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL" `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json"
```

## Direct prediction-file mode

If automatic search cannot locate the prediction file, specify it directly:

```powershell
python scripts\d17_g_plot_cs3d_interactive.py `
  --batch 5 `
  --battery 1 `
  --cycles all `
  --target cs_c `
  --pred_npz "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g3_frozen_test_report_only_audit/predictions/frozen_test/<YOUR_PRED_FILE>.npz" `
  --softlabel_root "E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL" `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json"
```

## Useful options

```text
--target cs_a / cs_c / both
--cycles 13-15 / 13,14,15 / all
--backend QtAgg / TkAgg       # set only if your Matplotlib default backend is not interactive
--save_png --save_dir <dir>   # save publication-draft PNG while still showing the popup
--max_plot_time_points 1200   # reduce if the popup is slow
--no_annotate_cycles          # hide cycle text labels
--no_sync_zlim                # independent z limits for prediction and truth
```

## Notes

- The script first tries to use `<target>_true_report_only` inside the prediction NPZ because this guarantees the same time-grid as the prediction.
- If truth is not in the prediction NPZ, it loads `solution_softlabels.npz` and aligns by time.
- If cycle IDs are not stored in the prediction/soft-label files, the script aligns cycle IDs from the replay profile using nearest-time matching.
- If cycle selection returns zero points, verify that the prediction file actually covers the requested cycles. Some D17-G prediction files were generated with `max_time_points=512` and/or a limited time window.
