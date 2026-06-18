# D17-P3.2 aggressive voltage recovery smoke

本包是 D17-P3.2 完整覆盖包，用于在 P3.1 已经机制通过但电压 MAE 仍约 0.118 V 的基础上，激进恢复电压反演能力，同时继续保持 D17 no-state-label 边界。

## 这次修了什么

1. 修复 P3.1 实质 bug：`voltage_inverse_residual_gate_mode` 写在 config 里，但旧 `p31_trainer` 没有传入 `D17MechanisticPINN`，实际仍使用默认 low/transition gate。
2. 新增 `all_bounded / non_rest_plus_transition / low_transition` 可配置 gate。
3. 新增 smooth voltage-basis residual：每个 profile 只优化少量平滑基函数系数，不允许逐点把 `V_pred` 直接设成 `V_exp`。
4. final summary 改为报告 best voltage-safe checkpoint，而不是最后 epoch。
5. 保留 hard inventory projection、zero-volume-mean delta、theta/cs bounds audit，不允许用破坏内部状态换电压。

## 覆盖文件

覆盖到项目根目录即可：

```text
gv1/d17_pinn/model.py
gv1/d17_pinn/losses.py
gv1/d17_pinn/p32_trainer.py
gv1/d17_pinn/__init__.py
scripts/d17_p32_aggressive_voltage_recovery_12profile.py
scripts/d17_p32_inspect_summary.py
configs/d17_pinn_rebuild_p32_12profile_voltage_recovery.json
configs/d17_pinn_rebuild_p32_12profile_voltage_recovery.yaml
```

包内也保留了 P2/P3/P3.1 依赖文件，避免漏文件导致 import error。

## 运行命令

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

python scripts\d17_p32_aggressive_voltage_recovery_12profile.py `
  --config configs/d17_pinn_rebuild_p32_12profile_voltage_recovery.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --resolved_spec "configs/resolved_p2dlite_spec_placeholder.json" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p32_12profile_aggressive_voltage_recovery" `
  --profile_count 12 `
  --max_time_points 512 `
  --time_window_s 40000 `
  --n_r 17 `
  --epochs 180 `
  --warmup_epochs 25 `
  --voltage_recovery_until_epoch 140 `
  --lr 0.001 `
  --device auto
```

## 查看结果

```powershell
python scripts\d17_p32_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/p32_12profile_aggressive_voltage_recovery/D17_P32_12PROFILE_VOLTAGE_RECOVERY_SUMMARY.json"
```

## 判别标准

优先看控制台输出：

```text
status
reasons
final_voltage_mae_mean_V
final_voltage_corr_mean
voltage_target_met
zero_mean_a_max / zero_mean_c_max
```

建议解释：

```text
status = PASS 且 voltage_target_met = true：可以进入 P3.3 扩到 24-profile smoke。
status = PASS 但 voltage_target_met = false：机制干净，电压有改善但未达到 0.08 V，继续 voltage recovery。
status = REVIEW：不要扩规模，把 summary 发回诊断。
```

## 不变的禁止项

P3.2 仍禁止训练读取：

```text
cs_a / cs_c / theta_a / theta_c / phie / phis_c soft labels
theta0_oracle / oracle_shift
validation/test state-label feedback
```

P3.2 只使用 replay profile 中的：

```text
t_global_s / I_profile / voltage_exp / temperature_C / protocol metadata
```

