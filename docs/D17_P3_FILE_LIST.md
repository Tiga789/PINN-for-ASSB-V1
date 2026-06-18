# D17-P3 package files

```text
gv1/d17_pinn/config.py                         # no-PyYAML config loader, carried forward
gv1/d17_pinn/latent_adapter.py                 # adds raw latent offset support
gv1/d17_pinn/model.py                          # adds bounded low/transition inverse voltage residual
gv1/d17_pinn/p3_trainer.py                     # 6-profile mechanism smoke trainer
gv1/d17_pinn/__init__.py                       # exports P3 trainer
scripts/d17_p3_mechanism_smoke_6profile.py     # P3 CLI
scripts/d17_p3_inspect_summary.py              # compact summary checker
configs/d17_pinn_rebuild_p3_6profile_smoke.json
configs/d17_pinn_rebuild_p3_6profile_smoke.yaml
README_D17_P3_6PROFILE_MECHANISM_SMOKE.md
```

Existing P2 files are included where needed so the package can be overlaid safely on the D17 branch.
