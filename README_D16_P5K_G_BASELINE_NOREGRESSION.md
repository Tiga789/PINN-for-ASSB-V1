# D16-P5K-G baseline-only no-regression audit

这一步不训练、不加载 checkpoint、不改模型。它只做 baseline-only 审计：

- 读取现有 P5K-F manifest；
- 读取 D15 ALL55 P2Dlite-RG soft-label；
- 用 P5K-C 与 P5K-F 的 hard baseline/output transform，令 raw residual = 0；
- 重新计算 theta/cs mean 的 exact MAE/R2；
- 输出一个 Markdown 报告，判断下一版 P5K-G 的 baseline 是否存在 no-regression 风险。

## 输出文件

默认输出：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kg_baseline_noregression_audit\D16_P5KG_BASELINE_NOREGRESSION_AUDIT_REPORT.md
```

把这个 `.md` 文件内容贴给我检查即可。

## Smoke

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kg_baseline_audit.ps1 `
  -AllowOverwrite `
  -LimitProfiles 6 `
  -ChunkSize 200000
```

检查：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_check_d16_p5kg_baseline_audit_outputs.ps1
```

## Full audit

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kg_baseline_audit.ps1 `
  -AllowOverwrite `
  -ChunkSize 200000
```

如果磁盘空间紧张，把 cache 指到大盘，并保留默认的每 profile 清理缓存策略：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kg_baseline_audit.ps1 `
  -AllowOverwrite `
  -ChunkSize 100000 `
  -MmapCacheRoot "F:\_p5kg_baseline_audit_mmap_cache"
```

默认会在每个 profile 处理后清理 mmap cache，防止再次出现 `No space left on device`。

## 关键判断

报告会比较：

- P5K-C baseline-only
- P5K-F baseline-only
- existing P5K-C/P5K-F final eval references
- eval / core_train / hard_probe / batch / protocol

如果 P5K-F baseline-only 已经不能超过 P5K-C baseline-only，或者 hard_probe baseline 仍然灾难性负 R²，则下一版不应直接长训练，而应继续修 profile-level theta0/OCP/cbar 初始化。
