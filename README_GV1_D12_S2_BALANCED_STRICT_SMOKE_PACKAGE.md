# GV1 D12-S2 balanced strict smoke package

Added files:

```text
scripts/gv1_d12_s2_prepare_balanced_strict_smoke_commands.py
scripts/run_gv1_d12_s2_prepare_balanced_strict_smoke_commands.ps1
scripts/run_gv1_d12_s2_preflight.ps1
scripts/run_gv1_d12_s2_run_triplet.ps1
scripts/gv1_d12_s2_scorecard_from_predictions.py
scripts/run_gv1_d12_s2_collect_scorecard.ps1
scripts/run_gv1_d12_s2_all_in_one_safe.ps1
docs/D12_S2_BALANCED_STRICT_SMOKE_README.md
```

This package assumes the D12 metadata runtime wrapper already exists:

```text
scripts/gv1_train_conditioned_pinn_d12_metadata_runtime.py
```

It does not modify `gv1/model.py`, `gv1/output_transform.py`, `gv1/profile_adaptive.py`, `gv1/losses.py`, `gv1/trainer.py`, or the D9.6/D9.5.1 mainline training script.
