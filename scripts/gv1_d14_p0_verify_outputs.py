#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify D14-P0 audit outputs and print a compact status table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict JSON: {path}")
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify D14-P0 freeze audit JSON")
    ap.add_argument("--audit-json", required=True, help="Path to D14_P0_FREEZE_AUDIT.json")
    ap.add_argument("--allow-warn", action="store_true", help="Exit 0 when overall status is WARN")
    args = ap.parse_args()

    path = Path(args.audit_json).expanduser()
    data = load_json(path)
    checks: List[Dict[str, Any]] = list(data.get("checks") or [])
    overall = str(data.get("overall_status", "UNKNOWN"))

    print(f"audit_json={path}")
    print(f"overall_status={overall}")
    print("\nchecks:")
    for c in checks:
        print(f"  {c.get('status','?'):>4}  {c.get('check_id','?'):>4}  {c.get('title','')}")

    if overall == "PASS":
        return 0
    if overall == "WARN" and args.allow_warn:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
