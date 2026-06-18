# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect D17-P3.2 aggressive voltage recovery summary")
    ap.add_argument("summary_json")
    args = ap.parse_args()
    d = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    fa = d.get("final_aggregate", {})
    ia = d.get("initial_aggregate", {})
    ba = d.get("best_aggregate", {}) or {}
    print(json.dumps({
        "status": d.get("status"),
        "reasons": d.get("reasons"),
        "protocol": d.get("protocol"),
        "profile_count": d.get("profile_count"),
        "best_epoch": d.get("best_epoch"),
        "initial_voltage_mae_mean_V": ia.get("voltage_mae_V_mean"),
        "best_voltage_mae_mean_V": ba.get("voltage_mae_V_mean"),
        "final_voltage_mae_mean_V": fa.get("voltage_mae_V_mean"),
        "final_forward_voltage_mae_mean_V": fa.get("forward_voltage_mae_V_mean"),
        "final_voltage_corr_mean": fa.get("voltage_corr_mean"),
        "V_residual_inverse_abs_mean": fa.get("V_residual_inverse_abs_mean_V_mean"),
        "V_residual_basis_abs_mean": fa.get("V_residual_basis_abs_mean_V_mean"),
        "V_residual_total_abs_max": fa.get("V_residual_total_abs_max_V_max"),
        "zero_mean_max_a": fa.get("zero_mean_max_abs_a_mol_m3_max"),
        "zero_mean_max_c": fa.get("zero_mean_max_abs_c_mol_m3_max"),
        "theta_a_min_min": fa.get("theta_a_min_min"),
        "theta_a_max_max": fa.get("theta_a_max_max"),
        "theta_c_min_min": fa.get("theta_c_min_min"),
        "theta_c_max_max": fa.get("theta_c_max_max"),
        "training_uses_state_softlabels": d.get("no_state_label_policy", {}).get("training_uses_state_softlabels"),
        "checkpoint_selection_uses_state_softlabels": d.get("no_state_label_policy", {}).get("checkpoint_selection_uses_state_softlabels"),
        "selected_profiles": [x.get("canonical_cell_uid") for x in d.get("selected_profiles", [])],
        "model_voltage_recovery_config": d.get("p32_model_voltage_recovery_config"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
