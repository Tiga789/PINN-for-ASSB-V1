# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect D17-P3.1 summary")
    ap.add_argument("summary_json")
    args = ap.parse_args()
    d = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    out = {
        "status": d.get("status"),
        "reasons": d.get("reasons"),
        "profile_count": d.get("profile_count"),
        "best_epoch": d.get("best_epoch"),
        "initial_voltage_mae_mean_V": d.get("voltage_recovery", {}).get("initial_voltage_mae_mean_V"),
        "final_voltage_mae_mean_V": d.get("voltage_recovery", {}).get("final_voltage_mae_mean_V"),
        "final_voltage_corr_mean": d.get("voltage_recovery", {}).get("final_voltage_corr_mean"),
        "voltage_target_met": d.get("voltage_recovery", {}).get("target_met"),
        "training_uses_state_softlabels": d.get("no_state_label_policy", {}).get("training_uses_state_softlabels"),
        "zero_mean_a_max": d.get("final_aggregate", {}).get("zero_mean_max_abs_a_mol_m3_max"),
        "zero_mean_c_max": d.get("final_aggregate", {}).get("zero_mean_max_abs_c_mol_m3_max"),
        "theta_a_range": [d.get("final_aggregate", {}).get("theta_a_min_min"), d.get("final_aggregate", {}).get("theta_a_max_max")],
        "theta_c_range": [d.get("final_aggregate", {}).get("theta_c_min_min"), d.get("final_aggregate", {}).get("theta_c_max_max")],
        "selected_profiles": [x.get("canonical_cell_uid") for x in d.get("selected_profiles", [])],
        "loss_scale_audit_json": d.get("outputs", {}).get("loss_scale_audit_json"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
