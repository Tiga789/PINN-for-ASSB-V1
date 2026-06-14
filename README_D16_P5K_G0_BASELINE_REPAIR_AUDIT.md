# D16-P5K-G0 baseline-repair audit

这一步不训练、不加载 checkpoint、不改模型，只做 baseline 修复诊断。

## 它解决什么

上一轮 P5K-G baseline-only audit 暴露出两件事：

1. P5K-F final 在普通 eval 上略好于 P5K-C，但 P5K-F baseline-only 反而明显劣于 P5K-C baseline-only；
2. audit 脚本有 cleanup bug，把 numpy array 当成 gc module 调 `.collect()`，产生 110 个假失败。

P5K-G0 做两件事：

- 修复 `.collect()` cleanup bug；
- 增加 `theta0_oracle` 诊断候选，用 soft-label 初始 theta_mean 对 baseline 做 per-profile 初始平移，判断 hard-probe 是否主要是 theta0/OCP phase 错位。

注意：`theta0_oracle` 使用 soft-label 内部状态，**只用于诊断，不是可部署 baseline，不可作为正式训练/评估 promotion 证据**。

## 输出文件

默认输出：

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kg0_baseline_repair_audit\D16_P5KG0_BASELINE_REPAIR_AUDIT_REPORT.md
```

完成后把这个 `.md` 文件内容贴给我。

## Smoke

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kg0_baseline_repair_audit.ps1 `
  -AllowOverwrite `
  -LimitProfiles 6 `
  -ChunkSize 200000
```

检查：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_check_d16_p5kg0_outputs.ps1
```

## Full audit

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kg0_baseline_repair_audit.ps1 `
  -AllowOverwrite `
  -ChunkSize 200000
```

如果磁盘空间紧张：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kg0_baseline_repair_audit.ps1 `
  -AllowOverwrite `
  -ChunkSize 100000 `
  -MmapCacheRoot "F:\_p5kg0_baseline_repair_mmap_cache"
```

默认会在每个 profile 后清理 mmap cache。

## 关键判断

主要看四组：

- `P5K-C-baseline`
- `P5K-F-baseline`
- `P5K-C-theta0_oracle`
- `P5K-F-theta0_oracle`

如果 `theta0_oracle` 能显著修 hard_probe，说明问题主要是 profile-level initial inventory / OCP phase；下一步应做 observed-only theta0 estimator，而不是继续调 residual 或堆 epoch。

如果 `theta0_oracle` 也不能修 hard_probe，说明不仅是 theta0，还存在 q-integral scale / current sign / capacity scale / radial projection 不一致。
