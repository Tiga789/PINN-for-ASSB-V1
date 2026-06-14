# D16-P5K Eval v3 Fix

This package does not retrain. It adds a separate eval-only runner and a separate evaluator filename so the old broken evaluator cannot be called by accident.

Fixes:
- Uses `scripts/gv1_d16_p5k_eval55_vs_softlabels_v3.py` explicitly.
- Resolves missing/stale `softlabel_npz` paths from `profile_id` and `SoftlabelRoot`.
- Uses a short SHA1 mmap cache directory.
- Supports `-LimitProfiles` for smoke testing.
- Outputs exact R2 fields in P5K scorecard and CSV files.

Recommended flow:
1. Run v3 smoke with `-LimitProfiles 2`.
2. If evaluated=2 and failure_count=0, run full v3 eval without `-LimitProfiles`.
