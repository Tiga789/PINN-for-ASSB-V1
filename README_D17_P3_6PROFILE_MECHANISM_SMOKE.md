# D17-P3 6-profile mechanism smoke

Purpose: verify that D17-PINN can run on 6 replay profiles with shared mechanism-heavy model, observed-only profile latent inversion, and no state soft-label training.

This package keeps the D17 boundary:

- loads only replay NPZ observed fields: `t_global_s`, `I_profile`, `voltage_exp`, `temperature_C`, and metadata;
- does not load or train against `cs_a/cs_c/theta_a/theta_c/phie/phis_c` soft-label arrays;
- uses soft-label paths in the split manifest only as future report-only audit paths;
- checkpoint selection uses voltage/physics smoke metrics, not state-label R2.

P3 additions over P2:

1. 6-profile protocol-balanced selection from the train split.
2. Shared `D17MechanisticPINN` over all selected profiles.
3. Profile-wise raw latent offsets optimized only through observed voltage and physics losses.
4. Bounded low/transition voltage inverse residual inspired by D12-S1K. It is gated and amplitude-limited; it is not a direct assignment of `V_pred=V_exp`.

## Run

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

python scripts\d17_p3_mechanism_smoke_6profile.py `
  --config configs/d17_pinn_rebuild_p3_6profile_smoke.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --resolved_spec "configs/resolved_p2dlite_spec_placeholder.json" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p3_6profile_mechanism_smoke" `
  --profile_count 6 `
  --max_time_points 512 `
  --time_window_s 40000 `
  --n_r 17 `
  --epochs 80 `
  --warmup_epochs 20 `
  --lr 0.0008 `
  --device auto
```

Inspect:

```powershell
python scripts\d17_p3_inspect_summary.py "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p3_6profile_mechanism_smoke/D17_P3_6PROFILE_SMOKE_SUMMARY.json"
```

## Output

```text
D17_P3_6PROFILE_SMOKE_SUMMARY.json
training_history.csv
selected_profiles.json
model/best_model_and_latents.pt
model/last_model_and_latents.pt
predictions/D17_P3_PROFILE_00_PRED_OBS_ONLY.npz ...
```

Status interpretation:

- `PASS`: mechanism smoke ran on 6 profiles, no state labels used, zero-mean audit OK, and voltage MAE improved sufficiently under broad smoke criteria.
- `REVIEW`: code and protocol ran, but voltage inversion is not yet good enough for expansion. This should trigger P3.1 loss/prior/wrapper refinement, not full training.
- `FAIL`: do not continue; fix data/source/protocol/runtime first.
