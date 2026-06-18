
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/d17_g5_inspect_release.py <D17_G5_FINAL_RELEASE_MANIFEST.json>")
        return 2
    p = Path(sys.argv[1])
    d = json.load(open(p, "r", encoding="utf-8"))
    print(json.dumps({
        "protocol": d.get("protocol"),
        "status": d.get("status"),
        "final_release_ready": d.get("final_release_ready"),
        "candidate_id": d.get("candidate_id"),
        "recommendation": d.get("recommendation"),
        "reasons": d.get("reasons"),
        "g0": d.get("prerequisites", {}).get("g0"),
        "g21": d.get("prerequisites", {}).get("g21"),
        "g3": d.get("prerequisites", {}).get("g3"),
        "g4": d.get("prerequisites", {}).get("g4"),
        "key_metrics": d.get("key_metrics"),
        "artifact_hash_count": d.get("artifact_hash_count"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
