# D18-S2 Hotfix Fast Resume

This emergency hotfix avoids repeating the slow replay-root indexing step after the first D18-S2 preflight has already produced its audit files.

It resumes from:

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_d18_s2_preflight_micro_smoke/d18_s2_preflight
```

It keeps hard blockers for split, frozen-test exclusion, flagged-probe exclusion, replay resolution, profile-role separation, and Battery-1/Battery-10 exact UID checks. It treats git HEAD mismatch, free-disk warning, and original audit-only density warnings as non-blocking for this bounded 8-epoch micro-smoke.

It does not enable formal S2 training.
