# D15-P5A file map

| 文件 | 覆盖位置 | 作用 |
|---|---|---|
| `configs/d15_p5a_all55_existing_model_transfer_config.json` | `configs/` | 默认 ALL55 路径、旧模型路径、评估 stride、阈值 |
| `scripts/d15_p5a_selftest.py` | `scripts/` | 基础依赖和 canonical ID 自测 |
| `scripts/d15_p5a_preflight.py` | `scripts/` | 检查 ALL55 soft labels 和旧模型路径 |
| `scripts/d15_p5a_existing_model_transfer_eval.py` | `scripts/` | 核心 transfer evaluation，不训练 |
| `scripts/d15_p5a_pack_review.py` | `scripts/` | 打包 JSON/CSV review zip |
| `scripts/d15_p5a_run_all.ps1` | `scripts/` | 一键运行入口 |
| `README_D15_P5A.md` | 根目录 | 使用说明 |
| `D15_P5A_FILE_MAP.md` | 根目录 | 文件映射 |
| `D15_P5A_MANIFEST.json` | 根目录 | 包清单 |

本包不包含任何 `gv1/` 文件，不会覆盖 `gv1/__init__.py`。
