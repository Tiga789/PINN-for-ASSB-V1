#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import json, sys
from pathlib import Path

def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/d17_g7s1_inspect_summary.py <D17_G7S1_SMALL_FULLCYCLE_SMOKE_SUMMARY.json>"); return 2
    p = Path(sys.argv[1]); d = json.load(open(p, encoding="utf-8")); c = d.get("compact_metrics") or {}
    out = {"protocol": d.get("protocol"), "status": d.get("status"), "selected_cycle_check_ready": d.get("selected_cycle_check_ready"), "s2_ready": d.get("s2_ready"), "recommendation": d.get("recommendation"), "best_epoch": d.get("best_epoch"), "fit_train_mean_r2": c.get("fit_train_mean_r2"), "fit_train_min_r2": c.get("fit_train_min_r2"), "internal_heldout_mean_r2": c.get("internal_heldout_mean_r2"), "internal_heldout_min_r2": c.get("internal_heldout_min_r2"), "validation_mean_r2": c.get("validation_mean_r2"), "validation_min_r2": c.get("validation_min_r2"), "worst_internal_target_profile": c.get("worst_internal_target_profile"), "worst_validation_target_profile": c.get("worst_validation_target_profile"), "summary_json": str(p), "next_required_checks": d.get("next_required_checks")}
    print(json.dumps(out, ensure_ascii=False, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
