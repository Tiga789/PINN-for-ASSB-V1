# GV1 D11-B regime feature distance audit package

This package implements **D11-B: regime feature distance audit** for `B1_2C battery-8`.

It is report-only. It reads existing XJTU/GV1 replay profiles, training-ready manifests, cycle-level manifests, and D10 policy artifacts. It does **not** change D9.6/D9.5.1 mainline code.

## Files

```text
scripts/gv1_d11_b_regime_feature_distance_audit.py
scripts/run_gv1_d11_b_regime_feature_distance_audit.ps1
manifests/d11_b_expected_outputs.json
docs/D11_B_SCOPE.md
README_GV1_D11_B.md
package_info.json
```

## Run

From the project root:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d11_b_regime_feature_distance_audit.ps1"
```

Optional plots:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d11_b_regime_feature_distance_audit.ps1" -MakePlots
```

Default output directory:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_b_regime_feature_distance_audit
```

## Main outputs

```text
D11_B_RECOMMENDATION.md
d11_b_regime_feature_distance_summary.json
d11_b_profile_feature_table.csv
d11_b_battery8_vs_b1_2c_peer_distance.csv
d11_b_battery8_top_distance_features.csv
d11_b_b1_2c_pairwise_distance_matrix.csv
```

## Expected verdicts

```text
d11_b_battery8_feature_distance_boundary_supported_keep_flagged
d11_b_battery8_feature_distance_weakly_supported_keep_flagged
d11_b_battery8_not_isolated_by_available_replay_features_keep_flagged_no_model_change
d11_b_inconclusive_missing_or_insufficient_feature_evidence
```

## Guardrails

- Keep `GV1 D9.6 / D9.5.1 trend-first warmup rare-regime` frozen.
- Keep `B1_2C battery-8` flagged/excluded while unresolved.
- Do not run direct 24-profile 200ks mainline claim from this audit.
- D11-C is allowed only as design-only metadata/flag ablation after manual review of D11-B evidence.
