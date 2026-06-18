# D17-G1.3 validation-aware generator surrogate

## Purpose

D17-G1.2 proved that the 12-profile train closed-set generator surrogate can reproduce the D15 P2Dlite-RG soft-label outputs at high precision, but validation profiles failed because the model still used train-profile-id conditioning. D17-G1.3 removes train-profile-id memorization and uses observed replay features instead.

## What this package changes

Files included:

```text
gv1/d17_g/g1_data.py        # patched soft-label/replay time alignment dependency
gv1/d17_g/g1_metrics.py     # metric dependency
gv1/d17_g/g13_model.py
gv1/d17_g/g13_trainer.py
scripts/d17_g13_validation_aware_surrogate.py
scripts/d17_g13_inspect_summary.py
configs/d17_g13_validation_aware_surrogate.json
```

It does not overwrite G0, G1, G1.1, G1.2, D17-P, or D17-P4 files.

## Main design

- Keeps G1.2 multi-head structure and dedicated phie head.
- Removes train profile-id embedding.
- Adds observed profile encoder features from replay `I(t), V(t), T(t)` aligned to the soft-label target grid.
- Adds profile-level summary features: voltage endpoints/statistics, current statistics, charge/discharge Ah, rest/charge/discharge fractions, early/late dV/dt, temperature statistics.
- Uses train-cell soft labels for training.
- Holds out a small subset of train profiles internally for checkpoint scoring.
- Keeps validation soft labels report-only.
- Does not read frozen-test soft labels.

## Run

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

python scripts\d17_g13_validation_aware_surrogate.py `
  --config configs/d17_g13_validation_aware_surrogate.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g13_validation_aware_surrogate" `
  --train_profile_count 12 `
  --validation_profile_count 3 `
  --max_time_points 512 `
  --time_window_s 40000 `
  --epochs 900 `
  --lr 0.0007 `
  --batch_size 1024 `
  --device auto
```

Inspect:

```powershell
python scripts\d17_g13_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g13_validation_aware_surrogate/D17_G13_VALIDATION_AWARE_SURROGATE_SUMMARY.json"
```

## Read the result

`status=PASS` means the fit-train generator surrogate still learned the training profiles.

`g2_ready=true` is stricter. It requires fit-train, train-internal heldout, validation report-only, and phie/phis_c validation gates to pass.

If `status=PASS` but `g2_ready=false`, do not enter G2. Use:

```text
D17_G13_PER_TARGET_PROFILE_METRICS.csv
D17_G13_PER_TARGET_AGGREGATE.csv
D17_G13_PROFILE_ENCODER_FEATURE_AUDIT.csv
```

to identify whether the remaining issue is profile encoder generalization, phie/phis_c, or a specific validation profile.
