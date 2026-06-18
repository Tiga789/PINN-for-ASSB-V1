# D17-G1 time-alignment patch v1

## Fix

This patch fixes the G1 loader failure:

```text
ValueError: cs_a: cannot orient (100000, 17) for n_time=2702101
```

Root cause: the first G1 loader used the full replay time axis as `n_time`, while the D15 soft-label arrays were generated on a shorter generator output grid.  The target grid must be defined by the soft-label state arrays, then replay `I(t)`, `V(t)`, and `T(t)` must be interpolated onto that grid.

## Files

```text
gv1/d17_g/g1_data.py
README_D17_G1_TIME_ALIGNMENT_PATCH_V1.md
docs/D17_G1_TIME_ALIGNMENT_PATCH_V1_FILE_LIST.txt
```

## What changed

1. Infer target length from soft-label state arrays: `cs/theta/phie/phis`.
2. Prefer soft-label time axis when it matches the target grid.
3. Interpolate replay observed `I_profile`, `voltage_exp`, and `temperature_C` onto the soft-label target grid.
4. Do not use `phis_c` as an observed voltage feature fallback, because it is also a supervised target.
5. Keep validation/frozen-test policy unchanged. This patch only fixes data alignment.

## Run

After overlaying this patch, rerun the same G1 command:

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

## Expected behavior

The previous orientation error should disappear. If a new error appears, it should now be downstream of data alignment, not due to replay/soft-label time length mismatch.
