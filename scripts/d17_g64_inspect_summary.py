from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/d17_g64_inspect_summary.py <D17_G64_P4D_PROVENANCE_AUDIT_SUMMARY.json>")
        return 2
    p = Path(sys.argv[1])
    d = json.loads(p.read_text(encoding="utf-8"))
    print(json.dumps({
        "protocol": d.get("protocol"),
        "status": d.get("status"),
        "provenance_ready": d.get("provenance_ready"),
        "recommendation": d.get("recommendation"),
        "blockers": d.get("blockers"),
        "selected_profile_count": d.get("selected_profile_count"),
        "evaluated_profile_count": d.get("evaluated_profile_count"),
        "profile_provenance_scores": d.get("profile_provenance_scores"),
        "outputs": d.get("outputs"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
