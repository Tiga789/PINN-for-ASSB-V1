# D18-S2 preflight + micro-smoke runbook

## Installation boundary

Install through the force installer supplied outside `payload`. Do not drag the `payload` directory into the project. The installer copies each payload member to its exact project-relative destination, backs up replaced files and verifies hashes after copying.

## Runtime sequence

1. Validate the reviewed D18-S0/S1 outputs and the explicit human-review token.
2. Load the locked D17 split manifest and G2 prediction manifest.
3. Resolve replay files by exact canonical UID. Substring matching is prohibited.
4. Audit all selected source cycles before sampling.
5. Build the micro casepack from 2 early, 2 middle and 2 late cycles per selected profile.
6. Keep 64 stratified points per cycle only for the micro-smoke view.
7. Fit concentration-to-theta scales and current-integral cbar slopes using fit-train profiles only.
8. Run eight tiny epochs with AMP and `torch.compile` disabled.
9. Select the checkpoint using fit-train plus internal-heldout loss only.
10. Evaluate validation profiles only after the checkpoint is fixed.

## Selected profiles

### Fit-train

```text
Batch-1_2C_battery-1
Batch-2_3C_battery-5
Batch-3_R2.5_battery-2
Batch-4_R3_battery-4
Batch-5_random_walk_battery-2
Batch-6_GEO_battery-1
```

### Internal-heldout

```text
Batch-1_2C_battery-3
Batch-6_GEO_battery-5
```

### Validation report-only

```text
Batch-2_3C_battery-10
Batch-5_random_walk_battery-3
```

Frozen test and Batch-1 2C battery-8 are not loaded.

## Hard stop conditions

The micro-smoke does not start when any of the following occurs:

- prior S0/S1 evidence is missing or has the wrong status;
- project HEAD does not start with `9d995eb`;
- a selected UID, role, replay or soft-label file cannot be resolved exactly;
- battery-1 and battery-10 resolve to the same replay;
- any selected source cycle has fewer than 96 source points;
- fit profiles do not cover all six protocols and both branches;
- a frozen-test/flagged profile is selected;
- free disk space is below 2 GiB;
- AMP, `torch.compile` or formal-training switches are enabled.

## Main output files

```text
D18_S2_PREFLIGHT_MICRO_SMOKE_OVERALL_SUMMARY.json
D18_S2_ARCHITECTURE_CONTRACT.json
D18_S2_ARCHITECTURE_SYNTHETIC_CHECK.json

d18_s2_preflight\D18_S2_PREFLIGHT_SUMMARY.json
d18_s2_preflight\D18_S2_EXACT_UID_AUDIT.csv
d18_s2_preflight\D18_S2_SELECTED_PROFILE_MANIFEST.csv
d18_s2_preflight\D18_S2_PER_CYCLE_SOURCE_COVERAGE.csv
d18_s2_preflight\D18_S2_CANONICAL_SPLIT_VIEW.json

d18_s2_micro_casepack\D18_S2_PER_CYCLE_SOURCE_COVERAGE.csv
d18_s2_micro_casepack\D18_S2_MICRO_CASEPACK_BUILD_SUMMARY.json
d18_s2_micro_casepack\profiles\D18_S2_MICRO_CASEPACK_SUMMARY.json

d18_s2_micro_smoke\D18_S2_MICRO_SMOKE_SUMMARY.json
d18_s2_micro_smoke\D18_S2_training_history.csv
d18_s2_micro_smoke\D18_S2_metrics_by_profile_state.csv
d18_s2_micro_smoke\D18_S2_metrics_summary.json
```

A `PASS_MICRO_SMOKE` or `REVIEW_MICRO_SMOKE` result still keeps formal S2 training blocked.
