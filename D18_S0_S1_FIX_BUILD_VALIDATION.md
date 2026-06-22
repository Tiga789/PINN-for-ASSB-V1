# D18-S0/S1-FIX Build Validation

- Overall: **PASS**
- Python AST / compileall: **PASS**
- Unit tests: **12/12 PASS**
- Synthetic end-to-end: **PASS**
- S0 pointwise concentration/theta bounds: **PASS**
- Explicit 8-case S1 coverage gate: **PASS in synthetic fixture**
- Strict JSON and empty-CSV schemas: **PASS**
- Fresh ZIP extraction / CRC / duplicate / path checks: **PASS**
- Training launched: **False**
- Go to S2: **False**

Formal real-data export was not run in the build container because the frozen D17 checkpoint and ALL55 cache are on the user workstation. The formal command uses those existing local artifacts and fails closed when any required path or split is missing.
