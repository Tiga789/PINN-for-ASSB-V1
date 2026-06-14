# D15-P5A｜ALL55 existing-model transfer evaluation

## 目的

D15-P5A 不训练新模型，也不改动 55-cell soft labels。它只做一件事：

```text
用 D15 已经成功过的旧模型，评估它们能否直接迁移到 ALL55 P2Dlite-RG soft-label 数据集。
```

默认评估两个已有模型：

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p2_rg_precision_benchmark
E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p3c_batch2_15cell_rg_nn_benchmark
```

默认 ALL55 soft labels：

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL
```

## 运行

复制到项目根目录后运行：

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\d15_p5a_run_all.ps1 -AllowOverwrite
```

默认 eval stride 为 64，目的是避免一次性处理 55 个大 profile 时内存过高。如果你想更快 debug：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p5a_run_all.ps1 -AllowOverwrite -EvalStride 256
```

如果你想更严格：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p5a_run_all.ps1 -AllowOverwrite -EvalStride 32
```

## 输出

默认输出目录：

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p5a_all55_existing_model_transfer_eval
```

默认审查包：

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p5a_results_for_review.zip
```

请把 review zip 上传给我检查。

## 重要边界

- D15-P5A 不是新训练。
- D15-P5A 不生成 soft labels。
- D15-P5A 不证明 held-out 泛化；它只是先看旧模型直接迁移能力。
- 对 closed-set one-hot 模型，未见过的 cell 默认使用 all-zero onehot，并按 seen / unseen 分组报告。
- 同时报告 raw theta 和 projected theta；projection 只用于判断边界处理能否缓解 raw theta 越界。

## 结果解释

如果旧模型在 ALL55 上表现很好：

```text
先考虑 fine-tune / light calibration，不一定要从零训练 55-cell unified NN。
```

如果旧模型明显失败：

```text
进入 D15-P5B：ALL55 unified NN training 或 batch/protocol-aware expert 设计。
```
