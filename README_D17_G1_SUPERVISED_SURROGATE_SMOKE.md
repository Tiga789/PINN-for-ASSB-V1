# D17-G1 supervised generator-surrogate smoke

This package starts the D17-G branch after full G0 generator-equivalence audit has passed.

G1 is intentionally **not** the strict no-state-label D17-P route. It is a train-cell soft-label supervised generator surrogate smoke:

- train split soft labels are allowed in loss;
- validation soft labels are not used for training;
- frozen-test soft labels are not touched;
- checkpoint selection is train-loss only in this first smoke;
- validation metrics are report-only, to decide whether to proceed to G2.

## Files

```text
gv1/d17_g/__init__.py
gv1/d17_g/g1_data.py
gv1/d17_g/g1_model.py
gv1/d17_g/g1_metrics.py
gv1/d17_g/g1_trainer.py
scripts/d17_g1_supervised_surrogate_smoke.py
scripts/d17_g1_inspect_summary.py
configs/d17_g1_supervised_surrogate_smoke.json
docs/D17_G1_FILE_LIST_ACTUAL.txt
README_D17_G1_SUPERVISED_SURROGATE_SMOKE.md
```

## Run

```powershell
python scripts\d17_g1_supervised_surrogate_smoke.py `
  --config configs/d17_g1_supervised_surrogate_smoke.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g1_supervised_surrogate_smoke" `
  --train_profile_count 12 `
  --validation_profile_count 3 `
  --max_time_points 512 `
  --time_window_s 40000 `
  --epochs 150 `
  --lr 0.001 `
  --batch_size 1024 `
  --device auto
```

Inspect:

```powershell
python scripts\d17_g1_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g1_supervised_surrogate_smoke/D17_G1_SUPERVISED_SURROGATE_SMOKE_SUMMARY.json"
```

## Interpreting results

```text
status = PASS
```
means the supervised train-cell smoke ran successfully and train-cell generator labels were learned above the smoke threshold.

```text
promotion_status = PASS and g2_ready = true
```
means validation report-only accuracy is good enough to proceed to D17-G2 held-out generator-surrogate expansion.

```text
status = PASS but promotion_status = REVIEW
```
means the implementation runs, but validation generalization is not yet strong enough. Do not proceed to G2 before inspecting profile metrics.
