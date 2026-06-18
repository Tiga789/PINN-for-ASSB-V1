# D17-G0 semantic classifier patch v2

This patch fixes the G0 smoke issue where `status=REVIEW` because `semantics_known_fraction=0.25` despite code scan and profile audit being operationally clean.

## What changed

The previous G0 audit was too conservative because it read only NPZ headers and sidecar summary fields. In D15-RG repair-from-source branches, important generator semantics may be stored as small scalar/string arrays inside `solution_softlabels.npz`, such as:

```text
source_flux_method_a / source_flux_method_c
radial_solver_version
source_p2dlite_v1_key_a / source_p2dlite_v1_key_c
phis_c_voltage_preserved_from_source
```

This patch now reads only those small scalar metadata arrays and never loads large `cs/theta/phie/phis` arrays. It also classifies D15-RG source-repair branches as generator-semantically known when the output convention is clear enough for D17-G1 implementation.

## Files overwritten

```text
gv1/d17_g/__init__.py
gv1/d17_g/generator_equivalence.py
scripts/d17_g0_generator_equivalence_audit.py
scripts/d17_g0_inspect_audit.py
configs/d17_g0_generator_equivalence_audit.json
```

## Smoke command

```powershell
python scripts\d17_g0_generator_equivalence_audit.py `
  --project_root "." `
  --config configs/d17_g0_generator_equivalence_audit.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --softlabel_root "E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit_smoke_v2" `
  --profile_limit 8
```

## Pass criteria

```text
status = PASS
g1_ready = true
semantics_known_fraction >= 0.75
missing_npz_count = 0
missing_required_key_profile_count = 0
```

If smoke passes, run full G0 with the same command but without `--profile_limit 8`.
