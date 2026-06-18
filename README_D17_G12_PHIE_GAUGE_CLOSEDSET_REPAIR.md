# D17-G1.2 phie/gauge/target-scaling closed-set repair

本包是 G1.2 最小修复包，不进入 G2，不改 D17-P/P4，不改 G0，不覆盖 G1 主训练脚本。它只新增 G1.2 闭集修复脚本，用来验证 G1.1 暴露的 `phie` / gauge / target scaling 问题能否在 12-profile train closed-set 上被修复。

## 为什么需要 G1.2

G1.1 已经证明：

- 单 profile overfit 能达到很高精度，说明 loader、soft-label 时间网格、target orientation 基本正确。
- 12-profile train closed-set 的 `theta/cs/phis_c` 已经接近可用，但 `phie` 明显拖垮 closed-set min R²。
- 当前不能进 G2，必须先修 `phie/gauge/target scaling`。

## 本包怎么参考 generator 代码

G0 / generator code scan 已确认：

- D15-P0/P3/P4B 的 RG repair branch 从 source P2Dlite soft labels 读取 `cs/theta/cbar/J`，再调用 `generate_rg_profile()` 修复径向场。
- RG repair branch 保留 source voltage / phi labels；`phie` 并不是 D17 重新计算出来的唯一电化学电势 gauge。
- D15-P4D full replay current-integral branch 中 `phis_c` 可近似来自 `voltage_exp`，`phie` 可近似来自 ohmic-current-like potential。

因此 G1.2 不再把 `phie` 和 `cs/theta` 混进同一个普通 head，而是：

1. 使用 multi-head surrogate：`theta/cs/phis_c/phie` 分头输出。
2. 给 `phie` 单独的 direct observed-feature path。
3. 给 train closed-set profile 一个 phie profile-gauge embedding。
4. 使用 group-balanced loss，避免 34 维 cs 压制 1 维 phie。
5. 输出 per-target / per-profile CSV，不再只看 mean R²。

## 覆盖文件

```text
gv1/d17_g/g12_model.py
gv1/d17_g/g12_trainer.py
scripts/d17_g12_phie_gauge_closedset_repair.py
scripts/d17_g12_inspect_summary.py
configs/d17_g12_phie_gauge_closedset_repair.json
docs/D17_G12_FILE_LIST_ACTUAL.txt
README_D17_G12_PHIE_GAUGE_CLOSEDSET_REPAIR.md
```

## 运行

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

python scripts\d17_g12_phie_gauge_closedset_repair.py `
  --config configs/d17_g12_phie_gauge_closedset_repair.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g12_phie_gauge_closedset_repair" `
  --train_profile_count 12 `
  --validation_profile_count 3 `
  --max_time_points 512 `
  --time_window_s 40000 `
  --epochs 750 `
  --lr 0.0008 `
  --batch_size 1024 `
  --device auto
```

检查：

```powershell
python scripts\d17_g12_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g12_phie_gauge_closedset_repair/D17_G12_PHIE_GAUGE_CLOSEDSET_REPAIR_SUMMARY.json"
```

## 判读

```text
status = PASS
recommendation = G1_2_CLOSEDSET_REPAIRED_RERUN_G1_WITH_VALIDATION
```

说明 12-profile train closed-set 的 target/profile R² 已达到门槛，可以回到 G1 重新做 validation-aware supervised surrogate，而不是直接跳 G2。

```text
status = REVIEW
```

说明即使 phie/gauge/head/scaling 修复后，12-profile train closed-set 仍不够；不要进 G2，打开 `D17_G12_PER_TARGET_PROFILE_METRICS.csv` 看剩余失败 target/profile。

## 重要边界

G1.2 使用 train-cell soft labels，这是 supervised generator-surrogate 路线；它不是 strict no-state-label inverse PINN。validation soft labels 只 report-only，frozen-test 不读取。
