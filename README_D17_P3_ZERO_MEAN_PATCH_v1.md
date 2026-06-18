# D17-P3 zero-mean / inventory-safe patch v1

This patch fixes the P3 REVIEW caused by anode zero-volume-mean violation.

Root cause: the previous P3 model projected `delta_c` to zero volume mean, but then clamped the full `cs=cbar+delta_c` field. When profile-wise latent adaptation pushed anode cbar near/outside the physical window, this post-hoc clamp changed the spherical average and broke `mean(cs)=cbar`.

Changes:

- `latent_adapter.py`: raw latent zero now maps to prior-centered theta0 / qeff, not the midpoint of a broad range.
- `model.py`: removes post-hoc full-field `cs` clamp; keeps `cs=cbar+zero-mean delta_c` exactly. OCP lookup still clamps surface theta internally for numerical safety.
- `losses.py`: adds `state_bounds_loss` as a differentiable replacement for full-field clamp.
- `p3_trainer.py`: selection and status now account for zero-mean and theta-bound audits.
- `configs/d17_pinn_rebuild_p3_6profile_smoke.json`: increases zero-mean/inventory/state-bound weights.

Run the same P3 command with the JSON config.
