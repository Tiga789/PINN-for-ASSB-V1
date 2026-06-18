# D17-G1.5R stratified internal-heldout / coverage repair

This package is a minimal G1.5R patch. It does not overwrite G1.4 model/trainer files. It adds a new trainer/script that reuses the G1.4 model and loss, but replaces the accidental “last N profiles” internal-heldout rule with a deterministic protocol-stratified split.

## Why this package exists

D17-G1.5 triage found that G1.4 repaired validation phie but failed internal-heldout. The worst internal-heldout item was `Batch-4_R3_battery-4 / phie`, and the split was coverage-biased: fit-train contained 2C/3C/R2.5 while internal-heldout contained R2.5/R3. G1.5R tests whether that failure is mainly a split coverage issue.

## Files

```text
gv1/d17_g/g15r_trainer.py
scripts/d17_g15r_stratified_heldout_repair.py
scripts/d17_g15r_inspect_summary.py
configs/d17_g15r_stratified_heldout_repair.json
docs/D17_G15R_FILE_LIST_ACTUAL.txt
README_D17_G15R_STRATIFIED_HELDOUT_REPAIR.md
```

## Run

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

python scripts\d17_g15r_stratified_heldout_repair.py `
  --config configs/d17_g15r_stratified_heldout_repair.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g15r_stratified_heldout_repair" `
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

## Inspect

```powershell
python scripts\d17_g15r_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g15r_stratified_heldout_repair/D17_G15R_STRATIFIED_HELDOUT_REPAIR_SUMMARY.json"
```

## Decision

Only enter G2 if:

```text
status = PASS
g2_ready = true
```

If `status=PASS` but `g2_ready=false`, do not enter G2. Read:

```text
D17_G15R_STRATIFIED_SPLIT_AUDIT.csv
D17_G15R_PER_TARGET_PROFILE_METRICS.csv
D17_G15R_PHIE_AUDIT.csv
```

This run still uses train-cell soft labels for supervised generator-surrogate training. Validation soft labels are report-only. Frozen-test is not used.
