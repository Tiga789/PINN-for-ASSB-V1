from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.d17_g.g64_provenance_audit import run_provenance_audit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D17-G6.4 P4D/GEO soft-label provenance audit. No training, no patching, no full-array load.")
    p.add_argument("--project_root", default=".")
    p.add_argument("--config", default="configs/d17_g64_p4d_provenance_audit.json")
    p.add_argument("--split_manifest", required=True)
    p.add_argument("--g0_profile_semantics_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--profile_contains", action="append", default=[], help="Profile selector, e.g. Batch-6_GEO_battery-2. Can be repeated.")
    p.add_argument("--hash_large", action="store_true", help="Hash full large npz files. Default only hashes small files or first 1MB of large files.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_provenance_audit(args)
    print(json.dumps({
        "status": summary.get("status"),
        "provenance_ready": summary.get("provenance_ready"),
        "recommendation": summary.get("recommendation"),
        "blockers": summary.get("blockers"),
        "selected_profile_count": summary.get("selected_profile_count"),
        "evaluated_profile_count": summary.get("evaluated_profile_count"),
        "elapsed_s": summary.get("elapsed_s"),
        "summary_json": summary.get("outputs", {}).get("summary_json"),
        "profile_details_json": summary.get("outputs", {}).get("profile_details_json"),
        "local_code_scan_json": summary.get("outputs", {}).get("local_code_scan_json"),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
