# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary_json")
    args = ap.parse_args()
    d = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    print(json.dumps({
        "status": d.get("status"),
        "reasons": d.get("reasons"),
        "profile_count": d.get("profile_count"),
        "best_epoch": d.get("best_epoch"),
        "initial_voltage_mae_mean_V": d.get("initial_aggregate", {}).get("voltage_mae_V_mean"),
        "final_voltage_mae_mean_V": d.get("final_aggregate", {}).get("voltage_mae_V_mean"),
        "final_voltage_corr_mean": d.get("final_aggregate", {}).get("voltage_corr_mean"),
        "training_uses_state_softlabels": d.get("no_state_label_policy", {}).get("training_uses_state_softlabels"),
        "selected_profiles": [x.get("canonical_cell_uid") for x in d.get("selected_profiles", [])]
    }, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()
