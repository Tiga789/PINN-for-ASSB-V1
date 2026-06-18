# D17-P2 Patch: remove PyYAML hard dependency

This patch fixes:

```text
ModuleNotFoundError: No module named 'yaml'
```

Files included:

```text
gv1/d17_pinn/config.py
configs/d17_pinn_rebuild_p2_smoke.json
configs/d17_pinn_rebuild_p2_smoke.yaml
```

`config.py` now falls back to a built-in minimal YAML parser for the D17 smoke config, so PyYAML is no longer required.

After overwriting, rerun the same P2 command. You can also use the JSON config explicitly:

```powershell
python scripts\d17_p2_smoke_train.py `
  --config configs/d17_pinn_rebuild_p2_smoke.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --resolved_spec "configs/resolved_p2dlite_spec_placeholder.json" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/smoke_1profile_p2" `
  --split train `
  --profile_index 0 `
  --max_time_points 512 `
  --time_window_s 40000 `
  --n_r 17 `
  --epochs 15 `
  --lr 0.001 `
  --device auto
```
