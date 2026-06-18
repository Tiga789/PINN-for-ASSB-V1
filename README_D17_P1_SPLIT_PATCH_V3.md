# D17-P1 split manifest patch v3

This package replaces only:

```text
scripts/d17_make_split_manifest.py
```

Fix scope:

- Keeps v2 fix for `Batch-1_2C_battery-8` flagged_probe.
- Adds robust replay matching for Batch-5 / Batch-6 remaining profiles.
- Handles protocol name variants: `random_walk`, `randomwalk`, `random`, `RW`, `GEO`, `geo`, `geometric`, `geometric_sequence`.
- Safely reads only replay metadata keys from `solution_replay_profile.npz`: `cell_uid`, `profile_id`, `batch`, `protocol`, `battery`, etc.
- Does not read any soft-label state arrays.
- Writes `d17_replay_match_debug.json` in the split output directory.
- Prints missing replay names directly if any remain.

Expected output after rerun:

```text
status = PASS
counts = train 39 / validation 7 / frozen_test 8 / flagged_probe 1
battery8_flagged = true
missing_replay_count_for_normal_splits = 0
```
