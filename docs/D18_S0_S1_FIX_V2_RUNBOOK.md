# D18-S0/S1-FIX V2 runbook

## Purpose

Run only the corrected D18-S0 architecture contract and D18-S1 dense-array diagnostic. The workflow must not start D18-S2 training.

## Installation

Use the force installer shipped outside the payload. It verifies the archive payload, backs up changed project files, copies with overwrite enabled, and then runs installed verification.

## Verification

```powershell
powershell -ExecutionPolicy Bypass -File .\VERIFY_D18_S0_S1_FIX_V2.ps1
```

The verification checks installed hashes, compiles only D18 package modules/scripts, runs the package-specific 12-test suite, and executes a synthetic end-to-end dry run.

## Formal diagnostic run

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN_D18_S0_S1_FIX_V2.ps1
```

Default output:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d18_fullcycle_fix
```

The expected safety flags are:

```text
training_launched=false
go_to_s2=false
```
