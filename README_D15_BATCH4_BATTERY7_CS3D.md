# D15 Batch-4 R3 battery-7 cs_a / cs_c 3D plotting script

用途：在 PyCharm 中打开可拖动视角的 Matplotlib 3D 弹窗，绘制 `Batch-4_R3_battery-7` 的 `cs_a` 和 `cs_c` 预测浓度曲面。

默认读取：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_d15p0_8cell
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p1_rg_closedset_nn_smoke\model\best_with_state.pt
```

默认输出窗口报告：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_batch4_battery7_cs3d_plots\D15_BATCH4_BATTERY7_CS3D_WINDOW_REPORT.json
```

## 运行

把 `scripts/d15_plot_batch4_battery7_cs3d.py` 放到项目根目录的 `scripts/` 下，然后在 PyCharm Terminal 或 PowerShell 中运行：

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
python scripts\d15_plot_batch4_battery7_cs3d.py
```

如果 PyCharm 没有弹出可旋转窗口，可以指定后端：

```powershell
python scripts\d15_plot_batch4_battery7_cs3d.py --backend Qt5Agg
```

或：

```powershell
python scripts\d15_plot_batch4_battery7_cs3d.py --backend TkAgg
```

## 手动指定 3 个 4-cycle 窗口

```powershell
python scripts\d15_plot_batch4_battery7_cs3d.py `
  --cycle-window 20:23 `
  --cycle-window 120:123 `
  --cycle-window 240:243
```

## 同时保存 PNG

```powershell
python scripts\d15_plot_batch4_battery7_cs3d.py --save-png
```

## 只看 soft-label 真值曲面，不走 NN 预测

```powershell
python scripts\d15_plot_batch4_battery7_cs3d.py --surface-source true
```

