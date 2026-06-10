# D15-P1 file map

| 文件 | 覆盖/新增位置 | 作用 |
|---|---|---|
| `configs/d15_p1_nn_smoke_config.json` | `configs/` | D15-P1 NN smoke 配置、训练超参、scorecard 阈值。 |
| `gv1/p2dlite_rg_nn/__init__.py` | `gv1/p2dlite_rg_nn/` | 新增 NN smoke package 标记。 |
| `gv1/p2dlite_rg_nn/utils.py` | `gv1/p2dlite_rg_nn/` | JSON/CSV/路径/随机种子工具。 |
| `gv1/p2dlite_rg_nn/data.py` | `gv1/p2dlite_rg_nn/` | 读取 D15-P0 RG soft labels，构建 closed-set 训练/验证数据。 |
| `gv1/p2dlite_rg_nn/model.py` | `gv1/p2dlite_rg_nn/` | 小型 residual MLP。禁用 AMP/compile，适配 GTX 1080 Ti。 |
| `gv1/p2dlite_rg_nn/metrics.py` | `gv1/p2dlite_rg_nn/` | phis_c/phie/theta/radial-gradient 指标和 scorecard 阈值。 |
| `gv1/p2dlite_rg_nn/train_eval.py` | `gv1/p2dlite_rg_nn/` | 标准化、预测、单 profile 评估工具。 |
| `scripts/d15_p1_selftest_nn_smoke.py` | `scripts/` | 无真实数据的工具自测。 |
| `scripts/d15_p1_preflight.py` | `scripts/` | 检查 D15-P0 RG 输出是否可用于 NN smoke。 |
| `scripts/d15_p1_train_rg_closedset_nn_smoke.py` | `scripts/` | 训练 closed-set NN smoke。 |
| `scripts/d15_p1_eval_rg_closedset_nn_smoke.py` | `scripts/` | 全 profile 评估 NN 对 RG 标签的复现。 |
| `scripts/d15_p1_collect_scorecard.py` | `scripts/` | 汇总最终 D15-P1 scorecard。 |
| `scripts/d15_p1_run_all.ps1` | `scripts/` | Windows PowerShell 一键运行。 |
| `README_D15_P1.md` | 项目根目录 | 使用说明。 |
| `D15_P1_FILE_MAP.md` | 项目根目录 | 文件映射。 |
| `D15_P1_MANIFEST.json` | 项目根目录 | 包清单和 hash。 |
