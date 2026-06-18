# D17-P2 package file list

- `configs/d17_pinn_rebuild_p2_smoke.yaml` / `configs/d17_pinn_p2_smoke.yaml`：P2 smoke 配置。
- `scripts/d17_p2_smoke_train.py`：兼容入口，调用 `gv1_train_d17_pinn_rebuild.py`。
- `scripts/d17_p2_synthetic_smoke.py`：不依赖项目数据的合成 smoke。
- `scripts/gv1_train_d17_pinn_rebuild.py`：D17-P2 真实 train split 1-profile smoke trainer。
- `scripts/gv1_eval_d17_pinn_rebuild.py`：预留 P2/P3 评估入口。
- `gv1/d17_pinn/config.py`：配置加载与 no-state-label 配置守卫。
- `gv1/d17_pinn/p2dlite_prior.py`：resolved P2Dlite prior 解析；缺失时允许 smoke defaults。
- `gv1/d17_pinn/torch_ops.py`：torch 版积分、径向权重、zero-mean projection 等。
- `gv1/d17_pinn/latent_adapter.py`：observed-only latent adapter。
- `gv1/d17_pinn/electrochem_closure.py`：OCP/BV/ohmic/gauge/voltage closure。
- `gv1/d17_pinn/model.py`：D17MechanisticPINN。
- `gv1/d17_pinn/losses.py`：voltage + physics losses，无 state label loss。
- `gv1/d17_pinn/trainer.py`：P2 smoke training utilities。
- `gv1/d17_pinn/evaluator.py`：report/eval utilities。
- `gv1/d17_pinn/dataset.py`、`spec_resolver.py`、`cbar_core.py`、`radial_fv_core.py`、`audits.py`：P1 observed-only 基础模块随包保留。
