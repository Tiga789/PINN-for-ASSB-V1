# D17-G0 Generator Equivalence Audit

D17-G0 是 D17-G 路线的第一步：**先审计 XJTU P2Dlite-RG soft-label generator 的真实代码语义**，再训练 generator surrogate。它不训练模型、不选择 checkpoint、不修改旧 D17/P3/P4 文件。

## 目的

D17 之前的 no-state-label inverse PINN 在 P4-mini 中显示：电压可以到 60–70 mV 量级，但 `theta_c/cs_c/phie` 与 generator soft labels 严重不一致。这说明当前模型学到的是 voltage-consistent state，而不是 generator-consistent state。

D17-G0 因此只做三件事：

1. 扫描本地 generator 代码，确认 D15 P2Dlite-RG 的关键实现路径；
2. 读取 split manifest 与 soft-label NPZ **header/summary**，不加载 52GB 大数组，整理每个 profile 的状态字段和语义来源；
3. 输出 D17-G1 需要逐代码复刻/直接复用的机制清单。

## 覆盖文件

```text
gv1/d17_g/__init__.py
gv1/d17_g/generator_equivalence.py
scripts/d17_g0_generator_equivalence_audit.py
scripts/d17_g0_inspect_audit.py
configs/d17_g0_generator_equivalence_audit.json
docs/D17_G0_FILE_LIST_ACTUAL.txt
README_D17_G0_GENERATOR_EQUIVALENCE_AUDIT.md
```

## 运行 smoke

先只审计 8 个 profile，检查路径是否正常：

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

python scripts\d17_g0_generator_equivalence_audit.py `
  --project_root "." `
  --config configs/d17_g0_generator_equivalence_audit.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --softlabel_root "E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit_smoke" `
  --profile_limit 8
```

检查：

```powershell
python scripts\d17_g0_inspect_audit.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit_smoke/D17_G0_GENERATOR_EQUIVALENCE_AUDIT.json"
```

## 正式 D17-G0 audit

smoke 正常后跑全量：

```powershell
python scripts\d17_g0_generator_equivalence_audit.py `
  --project_root "." `
  --config configs/d17_g0_generator_equivalence_audit.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --softlabel_root "E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit"
```

检查：

```powershell
python scripts\d17_g0_inspect_audit.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_GENERATOR_EQUIVALENCE_AUDIT.json"
```

## 判读标准

```text
status = PASS
```

表示：generator 文件扫描和 soft-label profile 语义审计没有阻断，可以进入 D17-G1。

```text
status = REVIEW
```

表示：不要进入 D17-G1，先看 `reasons`、`missing_generator_files`、`missing_npz_count`、`missing_required_key_profile_count` 或 `semantics_known_fraction`。

## 主要输出

```text
D17_G0_GENERATOR_EQUIVALENCE_AUDIT.json
D17_G0_GENERATOR_CODE_SCAN.json
D17_G0_SPLIT_MANIFEST_SUMMARY.json
D17_G0_PROFILE_SEMANTICS.csv
D17_G0_NPZ_HEADER_SAMPLES.json
D17_G0_RECOMMENDATIONS.md
```

## 注意

D17-G0 不读取 soft-label 数组内容，只读 NPZ header 和 summary JSON。因此它不做状态精度评估，也不会消耗大量内存。D17-G1 开始才进入 supervised generator surrogate：训练集 soft labels 可以进入 loss，validation/frozen-test soft labels 只能按预先声明的协议用于评估。
