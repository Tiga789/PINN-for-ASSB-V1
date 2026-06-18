# D17-P2 机制一致 PINN smoke 包

本包进入 D17-P2：实现一个可执行的 mechanism-heavy / voltage-informed inverse PINN smoke。它只读取 P1 manifest 中的 `replay_npz`，训练输入为 `t_global_s / I_profile / voltage_exp / temperature_C / metadata`，不读取 `softlabel_npz` 作为训练源，不使用 `cs/theta/phie/phis` soft labels 做 loss。

## 新增/覆盖文件

```text
configs/d17_pinn_rebuild_p2_smoke.yaml
configs/d17_pinn_p2_smoke.yaml
scripts/d17_p2_smoke_train.py
scripts/d17_p2_synthetic_smoke.py
scripts/gv1_train_d17_pinn_rebuild.py
scripts/gv1_eval_d17_pinn_rebuild.py

gv1/d17_pinn/__init__.py
gv1/d17_pinn/config.py
gv1/d17_pinn/p2dlite_prior.py
gv1/d17_pinn/torch_ops.py
gv1/d17_pinn/latent_adapter.py
gv1/d17_pinn/electrochem_closure.py
gv1/d17_pinn/model.py
gv1/d17_pinn/losses.py
gv1/d17_pinn/trainer.py
gv1/d17_pinn/evaluator.py

# 为保证覆盖后完整，包内也包含 P1 已有的 observed-only 基础模块：
gv1/d17_pinn/dataset.py
gv1/d17_pinn/spec_resolver.py
gv1/d17_pinn/cbar_core.py
gv1/d17_pinn/radial_fv_core.py
gv1/d17_pinn/audits.py
```

## P2 中写入的 generator 机理

```text
1. measured-current replay：实测 I(t) 作为边界输入。
2. I(t) -> surface flux：表面通量由 I、R、eps_s、V_electrode、F 闭合。
3. I(t) -> cbar hard baseline：平均库存由电流积分产生，不由 NN 自由输出。
4. cs_j(t,r) = cbar_j(t) + delta_c_j(t,r)。
5. delta_c_j zero-volume-mean projection：径向 residual 不允许改变总库存。
6. spherical diffusion residual：固相球形扩散残差。
7. surface-flux boundary residual：颗粒表面通量边界残差。
8. theta = cs_surface / csmax，再进入正负极 OCP。
9. inverse Butler-Volmer 过电位 + R_ohm + voltage offset 电压闭合。
10. phis_c / phie gauge layer：显式处理电势公共基准。
11. observed-only latent adapter：从 I/V/T/profile features 估计 theta0/Qeff/Ds/i0/Rohm/gauge 等 choice。
12. low/transition local residual expert：可选，默认 smoke 关闭，避免一开始污染 baseline。
```

## 先做纯代码 synthetic smoke

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
python scripts\d17_p2_synthetic_smoke.py
```

预期输出类似：

```text
{
  "status": "PASS",
  "best_loss": ...,
  "final_voltage_mae_V": ...
}
```

如果 synthetic smoke 失败，先不要跑真实 XJTU 数据。

## 运行真实 P2 smoke

P1 已经 PASS 后运行：

```powershell
python scripts\d17_p2_smoke_train.py `
  --config configs/d17_pinn_rebuild_p2_smoke.yaml `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --resolved_spec "configs/resolved_p2dlite_spec_placeholder.json" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/smoke_1profile_p2" `
  --split train `
  --profile_index 0 `
  --max_time_points 512 `
  --time_window_s 40000 `
  --n_r 17 `
  --epochs 15 `
  --lr 0.001 `
  --device auto
```

输出目录：

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/smoke_1profile_p2/
  D17_P2_SMOKE_SUMMARY.json
  D17_P2_SMOKE_PRED_OBS_ONLY.npz
  training_history.csv
  model/best_model.pt
  model/last_model.pt
```

## 通过标准

`D17_P2_SMOKE_SUMMARY.json` 中优先看：

```text
status = PASS 或 REVIEW
no_state_label_policy.training_uses_state_softlabels = false
no_state_label_policy.profile_loader = replay_npz observed-only
final_metrics.voltage_mae_V 为有限数
final_metrics.zero_mean_max_abs_a_mol_m3 / c_mol_m3 接近 0
```

P2 是 smoke，不追求 R²，也不使用 soft labels 计算 promotion。若出现 `D17-P2 refused profile because source contained state-answer keys`，说明 replay source 混入了 state arrays，应先修数据源再继续。

## P2 的边界

P2 不是正式训练。P2 只证明：

```text
dataset -> observed-only latent adapter -> cbar hard baseline -> radial residual -> OCP/BV/gauge -> voltage/physics loss
```

可以 forward/backward，并且没有把 `cs/theta/phie/phis` soft labels 当作答案。P2 通过后再进入 P3：多 profile 训练、validation no-drift、frozen-test report-only audit。
