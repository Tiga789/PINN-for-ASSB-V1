# D17-G1.4 phie validation robustness repair

## Purpose

G1.3 proved that the observed-profile-conditioned generator surrogate is the right direction, but it was not G2-ready because one validation target/profile, mainly `phie`, remained low. G1.4 is a focused repair:

- no train-profile-id embedding;
- no frozen-test soft labels;
- validation soft labels remain report-only;
- checkpoint selection uses only fit-train and train-internal-heldout metrics;
- phie gets a dedicated convention/gauge head with observed I/V/T/profile basis plus gated nonlinear residual;
- training coverage is increased by default to 24 train profiles with 4 train-internal-heldout profiles.

## Run

```powershell
python scripts\d17_g14_phie_validation_robustness.py `
  --config configs/d17_g14_phie_validation_robustness.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g14_phie_validation_robustness" `
  --train_profile_count 24 `
  --validation_profile_count 3 `
  --internal_heldout_count 4 `
  --max_time_points 512 `
  --time_window_s 40000 `
  --epochs 1000 `
  --lr 0.0006 `
  --batch_size 1024 `
  --device auto
```

Inspect:

```powershell
python scripts\d17_g14_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g14_phie_validation_robustness/D17_G14_PHIE_VALIDATION_ROBUSTNESS_SUMMARY.json"
```

## Decision

- `status=PASS` means G1.4 training and fit-train reproduction are healthy.
- `g2_ready=true` means G2 can start.
- `g2_ready=false` means do not enter G2; inspect `D17_G14_PHIE_ROBUSTNESS_AUDIT.csv` and `D17_G14_PER_TARGET_PROFILE_METRICS.csv`.

This stage is still a G1 repair stage, not G2 and not frozen-test evaluation.
