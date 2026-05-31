# D10-P5 Scope

D10-P5 is not a new training stage. It is the bridge between D10 evidence and D11 design.

## Inputs

- D10-P0 battery-8 regime/outlier judgement
- D10-P1 23-profile 200ks scorecard
- D10-P3 lightweight correction recommendation
- D10-P4 final mainline decision archive

## Output decision

The expected successful decision is:

```text
D9.6/D9.5.1 accepted for non-outlier 23-profile 200ks.
B1_2C battery-8 remains flagged/excluded.
No D10-P3 correction adopted.
Proceed to D11 design/audit only.
```

## D11 route priority

1. D11-A: report + flagged registry.
2. D11-B: regime feature-distance audit.
3. D11-C: small flag-aware metadata ablation, only after D11-B.
4. D11-D: late-2C discharge expert branch, only with holdout validation.

Forbidden:

- Hard voltage clamp or component clamp repair.
- Direct 24-profile 200ks mainline claim.
- Replacing D9.6/D9.5.1 with D9.6.1/D9.6.2/D9.6.3.
