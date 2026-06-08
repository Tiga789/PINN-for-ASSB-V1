# P2Dlite Prior File Usage After P4A

The standalone prior file remains the user-facing battery information file:

```text
configs/P2Dlite_prior_xjtu_lr18650la_v0.json
```

A user should edit this file, not Python source code, when applying the workflow
to another 18650 NMC/graphite cell. The same prior file should be read by:

- `scripts/gv1_generate_xjtu_p2dlite_softlabels.py`
- `scripts/gv1_audit_xjtu_p2dlite_softlabels.py`
- future XJTU P2Dlite PINN output-transform / physics-loss code
- future XJTU prediction/evaluation scripts

Each generated soft-label NPZ stores:

```text
resolved_spec_hash
```

If later model code reads a different prior file, its resolved hash should be
compared with the soft-label hash before training or evaluation proceeds.
