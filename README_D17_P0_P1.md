# D17-P0/P1 clean package

目标：先完成 D17-PINN重构的协议地基，不训练模型、不追求 R²、不覆盖旧主线。

本包只新增文件：

```text
scripts/d17_freeze_evidence.py
scripts/d17_make_split_manifest.py
scripts/d17_audit_no_state_label_inputs.py

gv1/d17_pinn/__init__.py
gv1/d17_pinn/dataset.py
gv1/d17_pinn/spec_resolver.py
gv1/d17_pinn/cbar_core.py
gv1/d17_pinn/radial_fv_core.py
gv1/d17_pinn/audits.py

configs/d17_pinn_rebuild_smoke.yaml
configs/d17_split_policy_seed20260615.json
configs/resolved_p2dlite_spec_placeholder.json
```

没有包含：

```text
gv1/__init__.py
main.py
util/*
integration_spm/*
旧 D9.6 / D12-S1K / ModelFin_112 文件
```

## P0：冻结 D16 / P5K-G / G4 起点证据

在项目根目录运行：

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

python scripts\d17_freeze_evidence.py `
  --project_root "." `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/00_freeze" `
  --result_path "E:/XJTU battery dataset/_gv1_cache/xjtu_d16_p5kg_rulev2_strict_gate_FAST/G_train12_rulev2_strict" `
  --force
```

预期输出：

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/00_freeze/
  d17_p0_freeze_manifest.json
  D17_P0_FREEZE_README.md
  copied_evidence/
```

通过条件：

```text
status = PASS
d17_p0_freeze_manifest.json 存在
git revision/status 已记录
旧结果只做 inventory，不被修改
```

## P1-a：生成固定 split manifest

先用 ALL55 soft-label 目录生成 cell-level split。脚本不会读取 `cs/theta/phie/phis` 数组，只扫描目录名和 summary JSON。

```powershell
python scripts\d17_make_split_manifest.py `
  --softlabel_root "E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL" `
  --replay_root "E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_replay_profiles" `
  --replay_root "E:/XJTU battery dataset/_gv1_cache/xjtu_batch2_replay_profiles_d15p3" `
  --replay_root "E:/XJTU battery dataset/_gv1_cache/xjtu_batch56_remaining14_replay_profiles_d15p4c" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split" `
  --seed 20260615 `
  --flag_cell "Batch-1_2C_battery-8" `
  --force
```

预期输出：

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/
  d17_split_manifest.json
  d17_split_manifest.csv
  d17_split_audit.json
```

通过条件：

```text
d17_split_audit.json 中 pass=true
train / validation / frozen_test 均非空
manifest_hash_sha256 已生成
Batch-1_2C_battery-8 进入 flagged_probe
```

说明：如果 `missing_replay_count_for_normal_splits > 0`，P1 仍可继续做协议审计，但 P2 训练前必须补齐对应 replay profiles 或重划只含 replay-ready 的 smoke split。

## P1-b：no-state-label 输入审计

```powershell
python scripts\d17_audit_no_state_label_inputs.py `
  --config configs/d17_pinn_rebuild_smoke.yaml `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --out_json "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/audit/no_state_label_audit.json" `
  --project_root "."
```

预期输出：

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/audit/no_state_label_audit.json
```

通过条件：

```text
pass = true
no_state_label_input_audit = true
config 中 state_supervised / softlabel_supervised / cs_soft / theta_soft / phie_soft / phis_c_soft 均为 false
checkpoint_selection.use_state_softlabel_metrics = false
```

## P1-c：可选 Python import smoke

```powershell
python - <<'PY'
from gv1.d17_pinn import radial_volume_weights, zero_volume_mean_project
import numpy as np
r = np.linspace(0, 1, 17)
d = np.random.randn(5, 17)
dp = zero_volume_mean_project(d, r)
w = radial_volume_weights(r)
print("zero_mean_max=", float(np.max(np.abs((dp*w).sum(axis=1)))))
PY
```

如果 PowerShell 不接受 heredoc，用下面一行：

```powershell
python -c "from gv1.d17_pinn import radial_volume_weights, zero_volume_mean_project; import numpy as np; r=np.linspace(0,1,17); d=np.random.randn(5,17); dp=zero_volume_mean_project(d,r); w=radial_volume_weights(r); print('zero_mean_max=', float(np.max(np.abs((dp*w).sum(axis=1)))))"
```

## P0/P1 完成后不要做的事

```text
不要训练 D17 模型。
不要用 d17_split_manifest 里的 softlabel_npz 做 training source。
不要看 frozen_test 的 cs/theta/phie/phis R² 后改 gate/rule。
不要覆盖 D9.6/D12-S1K/ModelFin_112 相关文件。
```

P2 才开始实现可微模型 forward/backward smoke。
