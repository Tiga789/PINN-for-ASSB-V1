# D16-P5A v7：full-profile 补齐缺失 cell（compact metrics 输出）

## 本轮问题复盘

前几版出现的问题：

1. **v1/v2/v3/v4 模型发现错误**：误把 D14/D12 的普通 `best.pt` 当成 D15-RG checkpoint。正确契约是 D15-P1/P2 的 `model/best_with_state.pt`。
2. **D15-P2 checkpoint 缺失**：本地没有展开的 `best_with_state.pt`，后来确认是 D15-P0 8-cell softlabel 输入目录缺失，D15-P2 没进入训练。
3. **ALL55 soft labels 本身是完备的**：最终目录是 `xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL`，不是 soft-label 缺失。
4. **v4 full-profile 只跑成 31/55**：缺失 profile 中出现 `MemoryError`，没有生成完整 55-cell scorecard。
5. **raw/projected/primary 三套 NPZ 重复保存**：浪费空间。主结果应只保留 `eval_full_profiles/predictions`。
6. **v6 仍失败**：为了保持 full-profile，v6 创建了超大 `y_true.npy` / `y_pred.npy` memmap，再打成单 NPZ；长 profile 会触发 `OSError(22, Invalid argument)`。

## v7 修复原则

v7 仍然是 **full-profile evaluation**，不是 sampled。它逐 chunk 遍历完整时间轴并计算 NN vs soft label 指标。

但 v7 不再为长 profile 保存完整 `y_true/y_pred` 大数组，而是：

- 读取已有 31 个 prediction 的 sidecar metrics 或 full NPZ metrics；
- 只补齐缺失 cell；
- 对缺失 cell 做 chunked full-profile inference；
- 直接流式累计全 profile 精度；
- 在同一个 `eval_full_profiles/predictions` 目录写入一个 **compact marker NPZ** 和 `.metrics.json`；
- 最终统一生成 55-cell metrics CSV / batch CSV / protocol CSV / scorecard。

这保证口径是 full-profile，不再 sampled，同时避免写 4GB+ 单个 NPZ 导致 Windows/NumPy 写盘失败。

## 使用方法

先补 2 个缺失 cell smoke：

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5a_fill_missing_v7.ps1 `
  -LimitMissing 2 `
  -Device "cuda:0" `
  -BatchSize 32768 `
  -ChunkSize 200000
```

检查：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_check_d16_p5a_v7_outputs.ps1
```

如果 count 从 31 增长到 33，再跑全部缺失：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5a_fill_missing_v7.ps1 `
  -Device "cuda:0" `
  -BatchSize 32768 `
  -ChunkSize 200000
```

如仍有内存问题，把 `ChunkSize` 降到 `100000` 或 `50000`。

## 输出路径

主 prediction 目录：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_V4_D15_existing_on_ALL55\eval_full_profiles\predictions
```

最终 scorecard：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_V4_D15_existing_on_ALL55\D16_P5A_V7_FINAL_SCORECARD.json
```

主要精度表：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_V4_D15_existing_on_ALL55\eval_full_profiles\D16_P5A_METRICS_BY_PROFILE.csv
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_V4_D15_existing_on_ALL55\eval_full_profiles\D16_P5A_BATCH_METRICS.csv
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_V4_D15_existing_on_ALL55\eval_full_profiles\D16_P5A_PROTOCOL_METRICS.csv
```

## 判断标准

最终目标：

```text
prediction npz count: 55
D16_P5A_V7_FINAL_SCORECARD.json 存在
operational_status = PASS 或 REVIEW
profile_count_with_metrics = 55
```

`REVIEW` 不等于脚本失败；通常表示某些 profile 指标未达阈值或有 failure 记录，需要看 `D16_P5A_FAILURES.json`。
