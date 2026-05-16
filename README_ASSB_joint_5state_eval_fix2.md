
# ASSB joint five-state evaluator fix2

This package only fixes the evaluator. It does not modify any training file.

Files:

```text
evaluate_assb_joint_states_soh.py
scripts/run_assb_joint_stageB_soh_state_eval.ps1
README_ASSB_joint_5state_eval_fix2.md
```

Main fix:

- `cs_a` / `cs_c` are no longer compared by flattening a sampled prediction array against the beginning of the full reference `solution.npz`.
- The evaluator now follows the 107A corrected evaluation NPZ convention and prefers:

```text
cs_a_true  vs cs_a_pred
cs_c_true  vs cs_c_pred
phie_true  vs phie_pred
phis_c_true vs phis_c_pred
```

- For concentration variables, `cycle_id_cs` is used for per-cycle metrics.
- For potential variables, `cycle_id_potential` is used for per-cycle metrics.
- If paired `*_true` / `*_pred` arrays do not exist, exact shape match or explicit nearest-time alignment is required. Silent truncation is refused.

Recommended command:

```powershell
.\scripts\run_assb_joint_stageB_soh_state_eval.ps1 `
  -PythonExe "D:\Anaconda\envs\torchgpu\python.exe" `
  -StageBEvalDir ".\EvalFin_110_stageB_aging" `
  -ReferenceNpz "..\assb_soft_labels_cycle5_522_v2_massclosed_candidate\solution.npz" `
  -StateEvalDir ".\EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only" `
  -OutputDir ".\EvalFin_110_joint_StageB_SOH_107A_states_fix2"
```

If auto-discovery fails, pass the 107A paired NPZ explicitly:

```powershell
D:\Anaconda\envs\torchgpu\python.exe .\evaluate_assb_joint_states_soh.py `
  --stageB_eval_dir ".\EvalFin_110_stageB_aging" `
  --reference_npz "..\assb_soft_labels_cycle5_522_v2_massclosed_candidate\solution.npz" `
  --state_prediction_npz ".\EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only\eval_sampled_arrays_ModelFin107A_csA_corrected.npz" `
  --cycle_table_csv ".\Data\assb_aging_fix1\cycle_table.csv" `
  --output_dir ".\EvalFin_110_joint_StageB_SOH_107A_states_fix2"
```

Key outputs:

```text
five_state_scorecard.csv
state_metrics_global.csv
state_array_alignment_provenance.json
state_npz_discovery.json
soh_stageB_metrics.json
joint_evaluation_summary.json
```

Check that `alignment_mode` is `paired_npz_internal` for `cs_a`, `cs_c`, `phie`, and `phis_c`.
