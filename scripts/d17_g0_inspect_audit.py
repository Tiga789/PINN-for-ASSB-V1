from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/d17_g0_inspect_audit.py <D17_G0_GENERATOR_EQUIVALENCE_AUDIT.json>")
        return 2
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"Missing file: {p}")
        return 2
    d = json.load(open(p, "r", encoding="utf-8"))
    pa = d.get("profile_audit", {})
    print("protocol=", d.get("protocol"))
    print("status=", d.get("status"))
    print("g1_ready=", d.get("g1_ready"))
    print("reasons=", d.get("reasons", []))
    print("code_scan_status=", d.get("code_scan", {}).get("status"))
    print("profile_audit_status=", pa.get("status"))
    print("profile_count_audited=", pa.get("profile_count_audited"))
    print("semantics_known_fraction=", pa.get("semantics_known_fraction"))
    print("semantic_known_profile_count=", pa.get("semantic_known_profile_count"))
    print("semantic_branch_counts=", pa.get("semantic_branch_counts"))
    print("phie_semantics_counts=", pa.get("phie_semantics_counts"))
    print("phis_c_semantics_counts=", pa.get("phis_c_semantics_counts"))
    print("profile_semantics_csv=", pa.get("profile_semantics_csv"))
    print("next_step=", d.get("next_step"))
    return 0 if d.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
