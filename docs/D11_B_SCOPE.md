# D11-B scope

D11-B quantifies whether `B1_2C battery-8` can be isolated by measured replay features before any new model changes.

The audit compares battery-8 to Batch-1 / 2C peers using:

- full-profile current, voltage, temperature, time-density, and cycle features;
- first 40 ks and first 200 ks windows;
- charge / discharge / rest segment features;
- cycle-level capacity and SOH features when the cycle manifest is available;
- robust peer z-scores and pairwise robust distances.

D11-B does not train a model and does not modify D9.6/D9.5.1 code.

The main decision boundary is:

```text
Strong feature separation -> keep battery-8 flagged; D11-C design-only metadata flag ablation may be planned after manual review.
Weak/no separation -> keep battery-8 flagged; do not proceed to D11-C model changes yet.
Inconclusive -> fix manifests/cache evidence first.
```
