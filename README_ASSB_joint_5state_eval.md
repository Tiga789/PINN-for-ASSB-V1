
# ASSB ModelFin110 joint five-state evaluation package

Purpose:

- Combine Stage-B SOH prediction with four electrochemical state evaluation.
- Five reported states: `cs_a`, `cs_c`, `phie`, `phis_c`, `SOH`.
- This is evaluation only. It does not train and does not open original `DATA_LOSS`.

Files:

```text
evaluate_assb_joint_states_soh.py
scripts/run_assb_joint_stageB_soh_state_eval.ps1
```

Recommended workflow:

1. Make sure Stage B has completed:

```powershell
Test-Path ".\EvalFin_110_stageB_aging\mechanism_by_cycle.csv"
```

2. Make sure you have a four-state prediction NPZ from the 107A/110 state evaluator. D4 notes that the 107A corrected eval directory contains npz outputs. The default wrapper searches:

```text
.\EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only
```

If the script cannot auto-find the correct `.npz`, pass it manually using `-StatePredictionNpz`.

3. Run:

```powershell
.\scripts\run_assb_joint_stageB_soh_state_eval.ps1 `
  -PythonExe "D:\Anaconda\envs\torchgpu\python.exe" `
  -StageBEvalDir ".\EvalFin_110_stageB_aging" `
  -ReferenceNpz "..\assb_soft_labels_cycle5_522_v2_massclosed_candidate\solution.npz" `
  -StateEvalDir ".\EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only" `
  -OutputDir ".\EvalFin_110_joint_StageB_SOH_107A_states"
```

Outputs:

```text
EvalFin_110_joint_StageB_SOH_107A_states\five_state_scorecard.csv
EvalFin_110_joint_StageB_SOH_107A_states\five_state_scorecard.json
EvalFin_110_joint_StageB_SOH_107A_states\joint_evaluation_summary.json
EvalFin_110_joint_StageB_SOH_107A_states\state_metrics_global.csv
EvalFin_110_joint_StageB_SOH_107A_states\state_metrics_by_cycle.csv
EvalFin_110_joint_StageB_SOH_107A_states\soh_stageB_metrics.json
EvalFin_110_joint_StageB_SOH_107A_states\soh_stageB_by_cycle.csv
EvalFin_110_joint_StageB_SOH_107A_states\five_state_nmae_scorecard.png
EvalFin_110_joint_StageB_SOH_107A_states\soh_stageB_pred_vs_obs.png
EvalFin_110_joint_StageB_SOH_107A_states\stageB_mechanism_profiles.png
```

If state prediction NPZ is not found:

- Check `state_npz_discovery.json` in the output folder.
- Re-run with an explicit prediction file, for example:

```powershell
D:\Anaconda\envs\torchgpu\python.exe .\evaluate_assb_joint_states_soh.py `
  --stageB_eval_dir ".\EvalFin_110_stageB_aging" `
  --reference_npz "..\assb_soft_labels_cycle5_522_v2_massclosed_candidate\solution.npz" `
  --state_prediction_npz ".\EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only\<YOUR_PREDICTION_FILE>.npz" `
  --output_dir ".\EvalFin_110_joint_StageB_SOH_107A_states"
```
