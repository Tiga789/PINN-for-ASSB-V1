from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from gv1.d17_g.g15_triage import run_g15_triage


def load_config(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="D17-G1.5 internal-heldout failure triage after G1.4")
    ap.add_argument("--config", required=False, default="configs/d17_g15_internal_heldout_failure_triage.json")
    ap.add_argument("--g14_out_dir", required=True, help="D17-G1.4 output directory")
    ap.add_argument("--g14_summary", default="", help="Optional explicit G1.4 summary JSON. Default: <g14_out_dir>/D17_G14_PHIE_VALIDATION_ROBUSTNESS_SUMMARY.json")
    ap.add_argument("--g13_summary", default="", help="Optional G1.3 summary JSON for comparison")
    ap.add_argument("--g12_summary", default="", help="Optional G1.2 summary JSON for comparison")
    ap.add_argument("--g0_profile_semantics_csv", default="", help="Optional G0 profile semantics CSV for annotation")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    g14_out_dir = Path(args.g14_out_dir)
    g14_summary = Path(args.g14_summary) if args.g14_summary else g14_out_dir / "D17_G14_PHIE_VALIDATION_ROBUSTNESS_SUMMARY.json"
    cfg = load_config(args.config)
    summary = run_g15_triage(
        g14_summary=g14_summary,
        g14_out_dir=g14_out_dir,
        out_dir=args.out_dir,
        config=cfg,
        g13_summary=Path(args.g13_summary) if args.g13_summary else None,
        g12_summary=Path(args.g12_summary) if args.g12_summary else None,
        g0_profile_semantics_csv=Path(args.g0_profile_semantics_csv) if args.g0_profile_semantics_csv else None,
    )
    print(json.dumps({
        "status": summary.get("status"),
        "recommendation": summary.get("recommendation"),
        "g2_ready": summary.get("g2_ready"),
        "g2_blockers": summary.get("g2_blockers"),
        "worst_internal_heldout": (summary.get("worst_internal_heldout_profiles") or [{}])[0],
        "summary_json": summary.get("files", {}).get("summary_json"),
        "decision_report_md": summary.get("files", {}).get("decision_report_md"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
