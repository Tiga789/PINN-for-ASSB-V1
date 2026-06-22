# D18-S1-FIX Explicit Dense Cases

The formal casepack is intentionally fixed and stratified. It does not search arbitrary `*.npz` files.

| Case | Split | Protocol | Purpose |
|---|---|---|---|
| Batch-2 3C battery-5, cycles 36–38 | train | 3C | known non-test dense failure |
| Batch-1 2C battery-3 | G2 internal-heldout | 2C | heldout cycle-history diagnosis |
| Batch-1 2C battery-7 | validation | 2C | validation 2C |
| Batch-2 3C battery-10 | validation | 3C | validation 3C |
| Batch-3 R2.5 battery-7 | validation | R2.5 | variable discharge protocol |
| Batch-4 R3 battery-2 | validation | R3 | partial-discharge protocol |
| Batch-5 random-walk battery-3 | validation | random_walk | P4D/current-integral branch |
| Batch-6 GEO battery-2 | validation | GEO | P4D/GEO branch |

Except for the known fixed cycle range, each case uses dense early, middle, and late cycle windows. The current D17 frozen-test and flagged-probe profiles are blocked by code and by the S1 coverage audit.
