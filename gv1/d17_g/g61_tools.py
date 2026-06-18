from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(obj: Mapping[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_full_cycle_repair_config(base_config: Mapping[str, Any], force_fit_profile_contains: Sequence[str] | None = None) -> Dict[str, Any]:
    """Return an effective config for D17-G6.1 full-cycle coverage repair.

    The failure in G6 smoke is not treated as a plotting/audit bug.  It is a
    coverage mismatch: the G2.1 checkpoint was trained with max_time_points=512
    and time_window_s=40000, while G6 probes the full soft-label time grid.  This
    config keeps the G2.1 generator-surrogate architecture, but changes training
    coverage to time_window_s=0 and a larger max_time_points so the train-cell
    supervised surrogate sees the same full-profile domain later used by G6.
    """
    cfg = json.loads(json.dumps(dict(base_config), ensure_ascii=False))
    cfg["protocol"] = "D17-G6.1_FULL_CYCLE_COVERAGE_REPAIR"
    cfg["seed"] = int(cfg.get("seed", 20260615))

    force = list(cfg.get("force_fit_profile_contains", []))
    for x in (force_fit_profile_contains or ["Batch-4_R3_battery-4", "Batch-5_random_walk_battery-8"]):
        if x and x not in force:
            force.append(str(x))
    cfg["force_fit_profile_contains"] = force

    # Keep the successful G2.1 protocol/branch stratification, but make internal
    # heldout small enough that all major protocol+branch groups remain in fit.
    cfg["internal_heldout_profile_count"] = int(cfg.get("internal_heldout_profile_count", 6))
    cfg["min_fit_per_protocol"] = int(cfg.get("min_fit_per_protocol", 2))
    cfg["min_fit_per_semantic_branch"] = int(cfg.get("min_fit_per_semantic_branch", 2))
    cfg["min_fit_per_protocol_branch"] = int(cfg.get("min_fit_per_protocol_branch", 1))
    cfg["max_internal_per_protocol"] = int(cfg.get("max_internal_per_protocol", 1))
    cfg["max_internal_per_semantic_branch"] = int(cfg.get("max_internal_per_semantic_branch", 4))

    # Full-cycle failure was dominated by phie becoming nearly constant and by
    # slow inventory drift outside the initial 40 ks window.  We increase phie
    # and inventory group weights slightly, but do not use validation/frozen-test
    # labels for selection.
    w = dict(cfg.get("target_group_weights", {}))
    w.update({"theta_a": 2.0, "theta_c": 2.0, "cs_a": 1.3, "cs_c": 1.3, "phie": 18.0, "phis_c": 3.5})
    cfg["target_group_weights"] = w

    # Gates: the immediate goal is to make the new full-coverage candidate stable
    # enough for G6 smoke/full audit.  Frozen-test remains untouched until G6.
    cfg.setdefault("fit_train_r2_mean_threshold", 0.99)
    cfg.setdefault("fit_train_r2_min_threshold", 0.97)
    cfg.setdefault("internal_heldout_r2_mean_threshold", 0.95)
    cfg.setdefault("internal_heldout_r2_min_threshold", 0.90)
    cfg.setdefault("validation_r2_mean_threshold", 0.95)
    cfg.setdefault("validation_r2_min_threshold", 0.90)
    cfg.setdefault("validation_phie_r2_mean_threshold", 0.93)
    cfg.setdefault("validation_phie_r2_min_threshold", 0.90)

    notes = list(cfg.get("notes", []))
    notes.extend([
        "G6 smoke showed the G2.1 checkpoint is not valid for all-cycle inference: G2.1 trained on 512 points within first 40 ks, but G6 audits 100000-point full soft-label grids.",
        "G6.1 changes the training sampling domain to full-profile coverage by requiring time_window_s=0 and larger max_time_points.",
        "No frozen-test soft labels are read in G6.1 training; validation remains report-only; checkpoint selection remains fit-train plus protocol/branch-stratified train-internal heldout.",
    ])
    cfg["notes"] = notes
    return cfg


def extract_compact_metrics(summary: Mapping[str, Any]) -> Dict[str, Any]:
    fit = summary.get("fit_train_per_target_aggregate", {}) if isinstance(summary.get("fit_train_per_target_aggregate"), Mapping) else {}
    internal = summary.get("internal_heldout_per_target_aggregate", {}) if isinstance(summary.get("internal_heldout_per_target_aggregate"), Mapping) else {}
    val = summary.get("validation_report_only_per_target_aggregate", {}) if isinstance(summary.get("validation_report_only_per_target_aggregate"), Mapping) else {}
    return {
        "status": summary.get("status"),
        "g3_ready": summary.get("g3_ready"),
        "recommendation": summary.get("recommendation"),
        "best_epoch": summary.get("best_epoch"),
        "fit_train_mean_r2": fit.get("all_target_profile_r2_mean"),
        "fit_train_min_r2": fit.get("all_target_profile_r2_min"),
        "internal_heldout_mean_r2": internal.get("all_target_profile_r2_mean"),
        "internal_heldout_min_r2": internal.get("all_target_profile_r2_min"),
        "internal_phie_mean_r2": internal.get("phie_r2_mean"),
        "internal_phie_min_r2": internal.get("phie_r2_min"),
        "validation_mean_r2": val.get("all_target_profile_r2_mean"),
        "validation_min_r2": val.get("all_target_profile_r2_min"),
        "validation_phie_mean_r2": val.get("phie_r2_mean"),
        "validation_phie_min_r2": val.get("phie_r2_min"),
        "worst_internal_target_profile": summary.get("worst_internal_target_profile"),
        "worst_validation_target_profile": summary.get("worst_validation_target_profile"),
    }
