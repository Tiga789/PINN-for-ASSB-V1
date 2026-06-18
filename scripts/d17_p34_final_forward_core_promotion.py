# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d17_pinn.config import cfg_get, load_config
from gv1.d17_pinn.p34_resolved_spec import ensure_p34_resolved_spec
from gv1.d17_pinn.p33_trainer import train_p33_forward_core_reliability


def main() -> None:
    ap = argparse.ArgumentParser(description="D17-P3.4 final forward-core promotion gate: resolved-spec alignment + forward-voltage recovery")
    ap.add_argument("--config", default="configs/d17_pinn_rebuild_p34_final_forward_core_promotion.json")
    ap.add_argument("--split_manifest", default=None)
    ap.add_argument("--base_resolved_spec", default=None, help="Optional existing generator/resolved prior JSON. If omitted or placeholder, P3.4 searches and builds one.")
    ap.add_argument("--resolved_spec", default=None, help="Optional final aligned spec output path or prebuilt spec path.")
    ap.add_argument("--softlabel_root", default=None, help="D15 ALL55 softlabel root used only to search prior/summary JSON; state arrays are never loaded.")
    ap.add_argument("--replay_search_root", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--profile_count", type=int, default=None)
    ap.add_argument("--validation_profile_count", type=int, default=None)
    ap.add_argument("--time_window_s", type=float, default=None)
    ap.add_argument("--max_time_points", type=int, default=None)
    ap.add_argument("--n_r", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--warmup_epochs", type=int, default=None)
    ap.add_argument("--voltage_recovery_until_epoch", type=int, default=None)
    ap.add_argument("--validation_adaptation_steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["d17_protocol_version"] = 34
    cfg["experiment_name"] = "d17_p34_final_forward_core_promotion"
    cfg.setdefault("paths", {})
    cfg.setdefault("train", {})
    cfg.setdefault("validation", {})
    cfg.setdefault("p34_spec_alignment", {})

    if args.split_manifest:
        cfg["paths"]["split_manifest"] = args.split_manifest
    if args.base_resolved_spec:
        cfg["paths"]["base_resolved_spec"] = args.base_resolved_spec
    if args.resolved_spec:
        # If p34_spec_alignment.enabled=true, this is used as the aligned output file.
        # If disabled, it is used as the fixed input spec.
        cfg["paths"]["resolved_spec"] = args.resolved_spec
        cfg["p34_spec_alignment"]["out_resolved_spec"] = args.resolved_spec
    if args.softlabel_root:
        cfg["paths"]["softlabel_root"] = args.softlabel_root
    if args.replay_search_root:
        cfg["paths"]["replay_search_root"] = args.replay_search_root
    train_overrides = ["profile_count", "time_window_s", "max_time_points", "n_r", "epochs", "warmup_epochs", "voltage_recovery_until_epoch", "lr", "device"]
    for name in train_overrides:
        val = getattr(args, name)
        if val is not None:
            cfg["train"][name] = val
    if args.validation_profile_count is not None:
        cfg["validation"]["profile_count"] = args.validation_profile_count
        cfg["p34_spec_alignment"]["validation_profile_count"] = args.validation_profile_count
    if args.profile_count is not None:
        cfg["p34_spec_alignment"]["profile_count"] = args.profile_count
    if args.max_time_points is not None:
        cfg["p34_spec_alignment"]["max_time_points"] = args.max_time_points
    if args.validation_adaptation_steps is not None:
        cfg["validation"]["adaptation_steps"] = args.validation_adaptation_steps

    out_dir = Path(args.out_dir or str(Path(str(cfg_get(cfg, "paths.output_root"))) / "p34_final_forward_core_promotion"))
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_info = ensure_p34_resolved_spec(cfg, out_dir)
    summary33 = train_p33_forward_core_reliability(cfg, out_dir)

    p34_summary = dict(summary33)
    p34_summary["protocol"] = "D17-P3.4_FINAL_FORWARD_CORE_PROMOTION"
    p34_summary["p34_resolved_spec_alignment"] = spec_info
    p34_summary["p34_goal"] = "final P3 gate before P4: aligned prior + forward core voltage target + residual budget target"
    p34_summary["p34_no_state_label_statement"] = {
        "state_softlabels_loaded_for_training": False,
        "state_softlabels_loaded_for_spec_alignment": False,
        "alignment_uses": "observed replay I(t), V(t), T(t), manifest metadata and prior JSON candidates only",
    }
    # Promotion is inherited from P3.3 gates but now interpreted as the final P3 gate.
    p34_summary["p4_ready"] = bool(p34_summary.get("status") == "PASS" and p34_summary.get("promotion_status") == "PASS")
    p34_summary["p4_blockers"] = list(p34_summary.get("promotion_reasons", [])) + list(p34_summary.get("reasons", []))
    p34_json = out_dir / "D17_P34_FINAL_FORWARD_CORE_PROMOTION_SUMMARY.json"
    p34_json.write_text(json.dumps(p34_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": p34_summary.get("status"),
        "promotion_status": p34_summary.get("promotion_status"),
        "p4_ready": p34_summary.get("p4_ready"),
        "p4_blockers": p34_summary.get("p4_blockers"),
        "out_dir": str(out_dir),
        "resolved_spec": cfg.get("paths", {}).get("resolved_spec"),
        "spec_voltage_fit_rmse_V": (spec_info.get("voltage_only_fit") or {}).get("rmse_V") if isinstance(spec_info.get("voltage_only_fit"), dict) else None,
        "train_corrected_voltage_mae_mean_V": p34_summary.get("voltage_recovery", {}).get("train_corrected_voltage_mae_mean_V"),
        "train_forward_voltage_mae_mean_V": p34_summary.get("voltage_recovery", {}).get("train_forward_voltage_mae_mean_V"),
        "validation_corrected_voltage_mae_mean_V": p34_summary.get("voltage_recovery", {}).get("validation_corrected_voltage_mae_mean_V"),
        "validation_forward_voltage_mae_mean_V": p34_summary.get("voltage_recovery", {}).get("validation_forward_voltage_mae_mean_V"),
        "forward_core_reliability_status": p34_summary.get("residual_budget_audit", {}).get("forward_core_reliability_status"),
        "residual_budget_status": p34_summary.get("residual_budget_audit", {}).get("residual_budget_status"),
        "summary_json": str(p34_json),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
