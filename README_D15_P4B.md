# D15-P4B — remaining-ready 18-cell P2Dlite-RG soft-label generation

本包用于处理 D15-P4A-fix 已确认 replay-ready 的 18 个剩余 XJTU cells：

- Batch-1: battery-2/4/5/6/7/8
- Batch-3: battery-1/2/3/4/5/8
- Batch-4: battery-1/3/4/5/6/8

本包只生成 soft labels 并做 radial-gradient audit，不训练 NN。

## 不会覆盖

不会覆盖：

- `gv1/__init__.py`
- 任何 `gv1/` 文件
- D15-P0/P1/P2/P3/P3B/P3C 旧输出
- P4A-fix 输出
- ASSB ModelFin_112 / D12-S1K / P4B/P5B/P5C 旧基线

## 运行

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
powershell -ExecutionPolicy Bypass -File scripts\d15_p4b_run_all.ps1
```

重跑本阶段输出目录：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p4b_run_all.ps1 -AllowOverwrite
```

如果内存压力大：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p4b_run_all.ps1 -AllowOverwrite -Workers 1
```

如果希望节省磁盘空间但速度较慢：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p4b_run_all.ps1 -AllowOverwrite -Workers 2 -SaveMode compressed
```

默认 `SaveMode=uncompressed` 是为了避免 `.npz` 大矩阵压缩耗时过长。

## 输出

默认输出：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_d15p4b_ready18
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4b_ready18_radial_audit
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4b_ready18_scorecard
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4b_results_for_review.zip
```

## 上传检查

运行结束后上传：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4b_results_for_review.zip
```

不要上传：

```text
solution_softlabels.npz
full output directory
```

## 通过后能说明什么

如果 P4B 通过，可以说明：

```text
D15-P4A-fix 中 replay-ready 的 18 个剩余 Batch-1/3/4 cells 已完成 P2Dlite-RG soft-label generation 和 radial-gradient audit。
```

仍不能说明：

```text
剩余全部 32 cells 已完成；Batch-5/6 剩余 14 cells 仍需先补 replay profiles。
这些 cs_a/cs_c 是实验真实内部状态。
held-out cell 泛化已证明。
```
