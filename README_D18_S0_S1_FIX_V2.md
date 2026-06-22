# D18-S0/S1-FIX V2

This package corrects the deployment collision caused by the legacy
`tests/test_d18_core.py` from the earlier D18-P0/S0/S1 package.

Key changes:

- installation is performed by a force-copy installer rather than by relying on Explorer merge behavior;
- package tests use the collision-free path `tests/test_d18_s0_s1_fix_v2.py`;
- verification runs only the package-specific test file, not every pre-existing repository test;
- the legacy `VERIFY_D18_S0_S1_FIX.ps1` and `RUN_D18_S0_S1_FIX.ps1` names are replaced with safe forwarders;
- D18-S2 training remains disabled.

After installation, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\VERIFY_D18_S0_S1_FIX_V2.ps1
powershell -ExecutionPolicy Bypass -File .\RUN_D18_S0_S1_FIX_V2.ps1
```
