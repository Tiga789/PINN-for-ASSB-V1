# D17-G1.1 closed-set alignment diagnostic

本包用于诊断 D17-G1 的低 R² 不是继续扩大训练能解决的问题，还是 loader / time-grid / target normalization / model capacity / generator branch 混学导致的问题。

## 为什么需要 G1.1

G0 已经确认 55 个 profile 的 generator 语义全部可识别：41 个 `D15-RG_REPAIR_FROM_SOURCE_SOFTLABEL_BRANCH`，14 个 `D15-P4D_FULL_REPLAY_CURRENT_INTEGRAL_BRANCH`。G1 已经能训练，但 train mean R² 约 0.836、validation report-only mean R² 约 0.904，不能进入 G2。G1.1 不进入 G2，不做 promotion，只做止损诊断。

## 这次如何参考 generator 代码

G1.1 会扫描本地 generator 代码文件，尤其是：

- `scripts/d15_p0_generate_p2dlite_rg_softlabels.py`
- `scripts/d15_p4d_full_generate_one_rg_softlabel.py`
- `gv1/p2dlite_rg/radial_solver.py`
- `gv1/p2dlite_rg/io_utils.py`
- `gv1/p2dlite_rg/data.py`

并检查这些语义是否存在：

- D15-P0 从 source P2Dlite v1 标签读取 `cs/theta/cbar/J`，再调用 `generate_rg_profile()` 生成 RG 径向场。
- D15-P0 保留 source voltage/phi labels，只替换径向状态和诊断项。
- D15-P4D 是 replay current-integral branch，包含 fixed theta initial、capacity/current integration、`phis_c ≈ V`、`phie ≈ ohmic I` 等语义。
- `radial_solver.py` 的 finite-volume RG core、`infer_surface_flux_from_cbar()`、volume weights 和 zero-mean/cbar preservation 是后续 surrogate 必须对齐的 deterministic generator core。

## 覆盖文件

```text
gv1/d17_g/g11_diagnostics.py
scripts/d17_g11_closedset_alignment_diagnostic.py
scripts/d17_g11_inspect_summary.py
configs/d17_g11_closedset_alignment_diagnostic.json
docs/D17_G11_FILE_LIST_ACTUAL.txt
README_D17_G11_CLOSEDSET_ALIGNMENT_DIAGNOSTIC.md
```

不覆盖 G1 trainer/model/data 主文件，不改 soft labels，不改 D17-P/P4 主线。

## 运行

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

python scripts\d17_g11_closedset_alignment_diagnostic.py `
  --config configs/d17_g11_closedset_alignment_diagnostic.json `
  --project_root "." `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g11_closedset_alignment_diagnostic" `
  --single_profile_count 1 `
  --train_profile_count 12 `
  --validation_profile_count 3 `
  --max_time_points 512 `
  --time_window_s 40000 `
  --device auto
```

检查：

```powershell
python scripts\d17_g11_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g11_closedset_alignment_diagnostic/D17_G11_CLOSEDSET_ALIGNMENT_DIAGNOSTIC_SUMMARY.json"
```

## 判读

```text
status = PASS
recommendation = G1_CORE_CAN_OVERFIT_TRAIN_DATA_RERUN_G1_WITH_STRONGER_CONFIG
```

说明单 profile 与 12-profile closed-set 都能过拟合，G1 主问题更可能是训练配置不足；可以回到 G1 强化训练后再看 validation。

```text
single_profile_overfit.status = REVIEW
```

说明 1 个 profile 都不能被当前 loader/model/normalization overfit；不要继续 G1/G2，先修数据对齐、target normalization 或模型输出。

```text
single PASS, closedset REVIEW
```

说明 loader 基本没坏，但 12-profile 训练容量、branch mixing、target scale 或模型结构不够；下一步应做 branch-specific head / profile encoder / generator-core latent surrogate，而不是直接扩大到 G2。

## 主要输出

```text
D17_G11_CLOSEDSET_ALIGNMENT_DIAGNOSTIC_SUMMARY.json
D17_G11_GENERATOR_CODE_SCAN.json
D17_G11_TARGET_NORMALIZATION_AUDIT.csv
D17_G11_FEATURE_NORMALIZATION_AUDIT.csv
D17_G11_TIME_GRID_ALIGNMENT_AUDIT.csv
D17_G11_CS_THETA_CONSISTENCY_AUDIT.csv
D17_G11_SINGLE_PROFILE_PER_TARGET_METRICS.csv
D17_G11_CLOSEDSET_PER_TARGET_METRICS.csv
runs/single_profile_overfit/*
runs/train_closedset_12profile/*
```

注意：脚本运行完成后即返回 exit code 0；`status=REVIEW` 是诊断结果，不是 Python 崩溃。是否继续由 summary 中的 `recommendation` 决定。
