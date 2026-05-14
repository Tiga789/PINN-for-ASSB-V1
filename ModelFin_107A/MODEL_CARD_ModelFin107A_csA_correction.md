# ModelFin_107A — anode cs_a/theta_a correction wrapper

ModelFin_107A is a post-hoc wrapper around ModelFin_106.

- Base model weights: `ModelFin_106/best.pt`
- Existing potential correction: linear-cycle common-mode gauge from ModelFin_106
- New correction: smooth residual correction for `cs_a`; `theta_a` is recomputed as `cs_a / cs_a,max`
- Unchanged variables: `phis_c`, `phie`, `theta_c`, `cs_c`
- Evaluation output: `C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1\EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only`

This wrapper is intended to test whether the remaining full-cycle error is a systematic anode-state residual rather than a failure of the positive-electrode or potential branches.
