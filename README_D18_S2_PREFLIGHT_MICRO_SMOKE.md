# D18-S2 preflight + micro-smoke

This package implements only the bounded transition from reviewed D18-S0/S1 evidence to a tiny D18-S2 architecture smoke. It does **not** enable formal D18-S2 training.

## What it checks

- exact canonical UID matching, including a hard battery-1 versus battery-10 collision guard;
- locked D17 G2 roles and D17 split integrity;
- no frozen-test or flagged-probe use;
- source soft-label point counts for each selected cycle, without downsampling;
- a separately labeled micro-smoke view capped at 64 points per cycle;
- six fit profiles spanning 2C, 3C, R2.5, R3, random-walk and GEO;
- both RG and P4D branch adapters;
- two internal-heldout profiles for checkpoint selection;
- two validation profiles used only after checkpoint selection;
- cycle-level GRU, within-cycle segmented GRU, bounded inventory correction, low-rank zero-mean radial states and dynamic potential heads;
- theta derived from cs, physical range checks and branch-gradient checks.

## Important boundary

The micro casepack uses generator state arrays and a teacher initial-cbar anchor. It is an architecture/data-pipeline smoke only. Every result is marked:

```text
formal_s2_training_eligible = false
go_to_formal_s2_training = false
```

The source-cycle grid and the micro-smoke sampled view are reported separately. A passing micro-smoke is not a full-cycle accuracy claim.

## Commands

After installation, from the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\VERIFY_D18_S2_PACKAGE.ps1
powershell -ExecutionPolicy Bypass -File .\RUN_D18_S2_PREFLIGHT_MICRO_SMOKE.ps1
```

Default output:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d18_s2_preflight_micro_smoke
```

Upload the complete output directory as a ZIP for review. Do not launch formal S2 training from the generated checkpoint.
