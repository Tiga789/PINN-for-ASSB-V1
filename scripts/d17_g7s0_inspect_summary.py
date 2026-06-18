#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inspect D17-G7-S0 summary."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary_json")
    args = ap.parse_args()
    p = Path(args.summary_json)
    d = json.loads(p.read_text(encoding="utf-8"))
    keep = {
        "protocol": d.get("protocol"),
        "status": d.get("status"),
        "s1_ready": d.get("s1_ready"),
        "recommendation": d.get("recommendation"),
        "blockers": d.get("blockers"),
        "profile_count_requested": d.get("profile_count_requested"),
        "profile_count_pass": d.get("profile_count_pass"),
        "profile_count_fail": d.get("profile_count_fail"),
        "max_time_points_per_profile": d.get("max_time_points_per_profile"),
        "coverage_gate": d.get("coverage_gate"),
        "files": d.get("files"),
        "elapsed_s": d.get("elapsed_s"),
    }
    print(json.dumps(keep, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
