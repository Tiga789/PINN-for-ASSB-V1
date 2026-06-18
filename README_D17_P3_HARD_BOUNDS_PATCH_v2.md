# D17-P3 hard inventory/bounds patch v2

This patch fixes the P3 REVIEW caused by `theta/cs physical bounds audit failed` after zero-mean was restored.

## What changed

- `gv1/d17_pinn/model.py`
  - Adds a hard observed-current inventory feasible projection for `theta0` before constructing `cbar`.
  - Replaces soft-only state bounds with a row-wise scaling of `zero-mean delta_c`.
  - Does **not** clamp full `cs`; therefore `mean(cs)=cbar` and `zero-volume-mean delta_c` remain preserved.
  - Keeps all mechanisms no-state-label: no `cs/theta/phie/phis` soft labels are read.

- `gv1/d17_pinn/p3_trainer.py`
  - Uses the prior electrode stoichiometry windows (`theta_min/theta_max`) for the P3 physical-bounds audit.
  - Writes the audit bounds into the summary JSON.

- `configs/d17_pinn_rebuild_p3_6profile_smoke.json`
  - Included unchanged from v1 for reproducibility.

## Expected result

After rerunning the same P3 command, the previous hard failures should be resolved:

```text
zero-volume-mean audit: pass
physical theta/cs bounds audit: pass
```

Voltage MAE can remain around 0.07-0.09 V in this placeholder-prior smoke. P3 pass is a mechanism-smoke criterion, not final accuracy.
