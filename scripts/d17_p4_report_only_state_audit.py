# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d17_pinn.config import cfg_get, load_config
from gv1.d17_pinn.p4_state_audit import run_p4_report_only_state_audit


def main() -> None:
    ap = argparse.ArgumentParser(description="D17-P4 frozen report-only internal-state audit")
    ap.add_argument("--config", default="configs/d17_pinn_rebuild_p4_report_only_state_audit.json")
    ap.add_argument("--candidate_p34_dir", default=None)
    ap.add_argument("--candidate_p34v_dir", default=None)
    ap.add_argument("--split_manifest", default=None)
    ap.add_argument("--softlabel_root", default=None, help="Report-only softlabel root; manifest paths are preferred when available.")
    ap.add_argument("--resolved_spec", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--no_state_label_audit", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--n_r", type=int, default=None)
    ap.add_argument("--max_time_points", type=int, default=None)
    ap.add_argument("--time_window_s", type=float, default=None)
    ap.add_argument("--adaptation_steps", type=int, default=None)
    ap.add_argument("--train_adaptation_steps", type=int, default=None)
    ap.add_argument("--validation_adaptation_steps", type=int, default=None)
    ap.add_argument("--frozen_test_adaptation_steps", type=int, default=None)
    ap.add_argument("--flagged_probe_adaptation_steps", type=int, default=None)
    ap.add_argument("--train_profile_limit", type=int, default=None)
    ap.add_argument("--validation_profile_limit", type=int, default=None)
    ap.add_argument("--frozen_test_profile_limit", type=int, default=None)
    ap.add_argument("--flagged_probe_profile_limit", type=int, default=None)
    ap.add_argument("--adaptation_lr", type=float, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg: Dict[str, Any] = load_config(args.config)
    cfg["d17_protocol_version"] = 4
    cfg["experiment_name"] = "d17_p4_report_only_state_audit"
    cfg.setdefault("paths", {})
    cfg.setdefault("p4", {})

    path_overrides = {
        "candidate_p34_dir": args.candidate_p34_dir,
        "candidate_p34v_dir": args.candidate_p34v_dir,
        "split_manifest": args.split_manifest,
        "softlabel_root": args.softlabel_root,
        "resolved_spec": args.resolved_spec,
        "checkpoint": args.checkpoint,
        "no_state_label_audit": args.no_state_label_audit,
    }
    for k, v in path_overrides.items():
        if v:
            cfg["paths"][k] = v
    p4_overrides = {
        "n_r": args.n_r,
        "max_time_points": args.max_time_points,
        "time_window_s": args.time_window_s,
        "adaptation_steps": args.adaptation_steps,
        "train_adaptation_steps": args.train_adaptation_steps,
        "validation_adaptation_steps": args.validation_adaptation_steps,
        "frozen_test_adaptation_steps": args.frozen_test_adaptation_steps,
        "flagged_probe_adaptation_steps": args.flagged_probe_adaptation_steps,
        "train_profile_limit": args.train_profile_limit,
        "validation_profile_limit": args.validation_profile_limit,
        "frozen_test_profile_limit": args.frozen_test_profile_limit,
        "flagged_probe_profile_limit": args.flagged_probe_profile_limit,
        "adaptation_lr": args.adaptation_lr,
        "device": args.device,
    }
    for k, v in p4_overrides.items():
        if v is not None:
            cfg["p4"][k] = v

    out_dir = Path(args.out_dir or str(Path(str(cfg_get(cfg, "paths.output_root", "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild"))) / "p4_report_only_state_audit"))
    out_dir.mkdir(parents=True, exist_ok=True)
    scorecard = run_p4_report_only_state_audit(cfg, out_dir)
    print(json.dumps({
        "status": scorecard.get("status"),
        "promotion_status": scorecard.get("promotion_status"),
        "p5_ready": scorecard.get("p5_ready"),
        "normal_frozen_test_profile_count": scorecard.get("normal_frozen_test_profile_count"),
        "promotion_reasons": scorecard.get("promotion_reasons"),
        "out_dir": str(out_dir),
        "scorecard_json": str(out_dir / "D17_P4_SCORECARD.json"),
        "state_profile_metrics_csv": str(out_dir / "D17_P4_STATE_AUDIT_PROFILE_METRICS.csv"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
