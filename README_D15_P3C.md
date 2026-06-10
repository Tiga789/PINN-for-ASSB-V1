# D15-P3C｜XJTU Batch-2 15-cell P2Dlite-RG applicability expansion

本包用于把 D15-P3/P3B 的 Batch-2 三代表电池验证扩展到 Batch-2 全部 15 个 cell。

## 功能

1. 复用 D15-P3 已生成的 Batch-2 replay profiles：
   `E:/XJTU battery dataset/_gv1_cache/xjtu_batch2_replay_profiles_d15p3`
2. 从 replay manifest 选择全部 15 个 PASS profile。
3. 为全部 15 个 Batch-2 cell 生成 P2Dlite-RG soft labels。
4. 对 15-cell soft labels 做 radial-gradient audit。
5. 训练 15-cell closed-set NN benchmark，默认使用 CUDA（如果 torch 可见 GPU）。
6. 用 D15-P3B 的 inference-time theta projection 修复 raw NN 的 theta 越界，并输出 raw vs projected 指标。
7. 自动打包 review zip。

## 默认输出

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_d15p3c_batch2_15cell
E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p3c_batch2_15cell_radial_audit
E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p3c_batch2_15cell_rg_nn_benchmark
E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p3c_batch2_15cell_boundary_projection_repair
E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p3c_batch2_15cell_applicability_scorecard
E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p3c_results_for_review.zip
```

## 运行

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
powershell -ExecutionPolicy Bypass -File scripts\d15_p3c_run_all.ps1
```

如果需要覆盖 D15-P3C 自己的旧输出：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p3c_run_all.ps1 -AllowOverwrite
```

快速 debug：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p3c_run_all.ps1 -AllowOverwrite -Quick
```

正式结果不要用 `-Quick`。

只生成 15-cell soft labels 和 radial audit，先跳过 NN：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p3c_run_all.ps1 -AllowOverwrite -SkipNN
```

## GPU 说明

soft-label generator 和 radial audit 主要是 numpy / I/O 任务，不会充分吃 GPU；NN train/eval/projection inference 会默认使用 CUDA。默认 NN 配置：

```text
batch_size = 32768
hidden_dim = 320
num_hidden_layers = 5
epochs = 1800
eval_stride = 4
projection inference batch_size = 262144
```

如果显存不足，先用 `-Quick` 验证流程；正式 benchmark 用默认配置。

## 检查文件

运行结束后上传：

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p3c_results_for_review.zip
```

不要上传大文件：

```text
model/*.pt
solution_softlabels.npz
prediction arrays
```

## 边界

D15-P3C 是 Batch-2 15-cell closed-set applicability expansion。即使 PASS，也只能说明 Batch-2 的 P2Dlite-RG generator、radial audit 和 projection-repaired NN closed-set benchmark 成立；不能说明 held-out cell 泛化，也不能说明 cs_a / cs_c 是实验真实径向内部状态。
