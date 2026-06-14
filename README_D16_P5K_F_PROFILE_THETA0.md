# D16-P5K-F profile-theta0 hard-cbar/OCP residual package

Purpose: rollback to the P5K-C hard-cbar/OCP residual architecture, add an observed-only profile-level theta0/OCP initializer, and keep generator/P2Dlite-RG prior information as audit/weak no-regression context rather than a hard output anchor.

Training boundary remains strict:

- training reads only `t_global_s`, `I_profile`, `voltage_exp`;
- no `theta_a/theta_c/cs_a/cs_c/phie/phis_c` soft-label data loss is used;
- soft-label internal arrays are used only in baseline preflight and final evaluation.

New split structure:

- `core_train`: representative and non-extreme anchors;
- `hard_probe`: known hard profiles included in training exposure but reported separately;
- `eval`: held-out profiles.

Default train set is `F_train12_profile_theta0`: 8 core_train + 4 hard_probe + 43 eval.

## Commands

Build manifest:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kf_train_fast.ps1 `
  -AllowOverwrite `
  -BuildManifestOnly `
  -TrainSet "F_train12_profile_theta0"
```

Baseline-only preflight smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kf_train_fast.ps1 `
  -AllowOverwrite `
  -BaselineOnlyAuditOnly `
  -TrainSet "F_train12_profile_theta0" `
  -LimitProfiles 4 `
  -ChunkSize 200000
```

Smoke training:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kf_train_fast.ps1 `
  -AllowOverwrite `
  -TrainOnly `
  -TrainSet "F_train12_profile_theta0" `
  -Epochs 50 `
  -Device "cuda:0" `
  -BatchSize 131072 `
  -ValEvery 5
```

Full training + eval:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kf_train_fast.ps1 `
  -AllowOverwrite `
  -TrainSet "F_train12_profile_theta0" `
  -Epochs 1300 `
  -Device "cuda:0" `
  -BatchSize 131072 `
  -EvalBatchSize 65536 `
  -ChunkSize 200000 `
  -ValEvery 10
```

Check outputs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_check_d16_p5kf_outputs.ps1 `
  -TrainSet "F_train12_profile_theta0"
```

Output root:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kf_train12_profile_theta0_hard_cbar_FAST\F_train12_profile_theta0
```

Promotion target versus P5K-C:

- eval theta_a_mean_mae < 0.1506
- eval theta_c_mean_mae < 0.1343
- eval theta_a_mean_r2 > 0.4239
- eval theta_c_mean_r2 > 0.3250
- phis_c_r2 > 0.99

If baseline-only preflight is worse than P5K-C by a large margin, stop before long training and inspect profile theta0/OCP initializer.
