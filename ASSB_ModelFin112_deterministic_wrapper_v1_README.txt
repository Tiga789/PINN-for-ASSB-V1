ASSB ModelFin_112 deterministic wrapper/evaluator v1

目的
----
把 frozen ModelFin_107A 四状态来源 + ModelFin_112_deterministicSOH_ridge_g4 deterministic ridge SOH head
打包成一个 ModelFin_112_deterministic_wrapper 工程统一目录，并输出五目标 scorecard。

重要边界
--------
1. 这是工程统一 wrapper，不是端到端联合训练的单神经网络。
2. 四状态来自 frozen 107A state source；SOH 来自 deterministic ridge head。
3. ridge SOH 的 scaler/alpha/系数选择只用 train/val；test 只在模型固定后评估。
4. 不再使用 neural SOH head，不再跑多 seed 训练。

覆盖文件
--------
util/assb112_deterministic_wrapper.py
scripts/build_ModelFin112_single_model.py
scripts/build_ModelFin112_deterministic_wrapper.py
evaluate_ModelFin112_unified_5targets.py
evaluate_ModelFin112_deterministic_5targets.py
scripts/run_ModelFin112_deterministic_wrapper_eval.ps1

推荐运行
--------
Set-Location "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
.\scripts\run_ModelFin112_deterministic_wrapper_eval.ps1 -Clean

如果自动找不到四状态 scorecard/NPZ，手动指定：
.\scripts\run_ModelFin112_deterministic_wrapper_eval.ps1 -Clean `
  -StateScorecardCsv ".\EvalFin_111_seed42locked_repro_c00\five_state_scorecard.csv"

输出
----
ModelFin_112_deterministic_wrapper\unified_config.json
ModelFin_112_deterministic_wrapper\build_audit.json
EvalFin_112_deterministic_wrapper\five_state_scorecard.csv
EvalFin_112_deterministic_wrapper\five_target_compact_summary.csv
EvalFin_112_deterministic_wrapper\unified_eval_audit.json
