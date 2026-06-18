# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/d17_p4mini_inspect_summary.py <D17_P4MINI_ADAPTATION_DIAGNOSTIC_SUMMARY.json>")
        raise SystemExit(2)
    p = Path(sys.argv[1])
    d = json.loads(p.read_text(encoding="utf-8"))
    diag = d.get("diagnostic", {})
    long_run = d.get("runs", {}).get("long", {})
    short_run = d.get("runs", {}).get("short", {})
    print(json.dumps({
        "status": d.get("status"),
        "decision": diag.get("decision"),
        "recommendation": diag.get("recommendation"),
        "short_state_r2": short_run.get("state_r2"),
        "long_state_r2": long_run.get("state_r2"),
        "r2_delta_long_minus_short": diag.get("r2_delta_long_minus_short"),
        "long_step_r2_mean": diag.get("long_step_r2_mean"),
        "long_step_r2_min": diag.get("long_step_r2_min"),
        "failed_min_recovery_keys": diag.get("failed_min_recovery_keys"),
        "failed_target_keys": diag.get("failed_target_keys"),
        "long_voltage": long_run.get("voltage"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
