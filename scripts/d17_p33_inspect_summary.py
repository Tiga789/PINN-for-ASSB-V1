# -*- coding: utf-8 -*-
from __future__ import annotations
import json, sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/d17_p33_inspect_summary.py <D17_P33_FORWARD_CORE_RELIABILITY_SUMMARY.json>")
        raise SystemExit(2)
    p = Path(sys.argv[1])
    d = json.loads(p.read_text(encoding="utf-8"))
    vr = d.get("voltage_recovery", {})
    rb = d.get("residual_budget_audit", {})
    fa = d.get("formula_alignment_audit", {})
    print(json.dumps({
        "status": d.get("status"),
        "reasons": d.get("reasons"),
        "promotion_status": d.get("promotion_status"),
        "promotion_reasons": d.get("promotion_reasons"),
        "train_profile_count": d.get("train_profile_count"),
        "validation_profile_count": d.get("validation_profile_count"),
        "best_epoch": d.get("best_epoch"),
        "train_corrected_voltage_mae_mean_V": vr.get("train_corrected_voltage_mae_mean_V"),
        "train_forward_voltage_mae_mean_V": vr.get("train_forward_voltage_mae_mean_V"),
        "validation_corrected_voltage_mae_mean_V": vr.get("validation_corrected_voltage_mae_mean_V"),
        "validation_forward_voltage_mae_mean_V": vr.get("validation_forward_voltage_mae_mean_V"),
        "corrected_target_met": vr.get("corrected_target_met"),
        "forward_core_reliability_status": rb.get("forward_core_reliability_status"),
        "residual_budget_status": rb.get("residual_budget_status"),
        "d12_formula_migrated": fa.get("d12_s1k_transition_fade_formula_migrated"),
        "uses_placeholder_spec": fa.get("uses_placeholder_spec"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
