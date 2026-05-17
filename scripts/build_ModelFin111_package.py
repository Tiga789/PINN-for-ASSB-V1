# -*- coding: utf-8 -*-
r"""Build and verify the ModelFin_111 package metadata.

ModelFin_111 protects ModelFin_107A: four electrochemical states are referenced
from the frozen 107A state engine/evaluation outputs, while only the cycle-level
SOH head is newly trained.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.assb111_split import sha256_file, sha256_jsonable, load_manifest


def _json_clean(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_json_clean(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        val = float(x)
        return None if not math.isfinite(val) else val
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    return x


def _dir_file_hashes(path: Path, *, max_files: int = 2000) -> Dict[str, str]:
    if not path.exists() or not path.is_dir():
        return {}
    out: Dict[str, str] = {}
    for f in sorted(p for p in path.rglob("*") if p.is_file())[:max_files]:
        out[f.relative_to(path).as_posix()] = sha256_file(f)
    return out


def _read_json(path: Any) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        p = Path(path)
    except Exception:
        return {}
    # Some callers pass an empty value or "." for optional JSON files.  "." is a
    # directory, not a JSON file; skip it instead of raising PermissionError.
    if str(p).strip() in {"", "."}:
        return {}
    if (not p.exists()) or p.is_dir():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(dict(obj)), f, ensure_ascii=False, indent=2, sort_keys=True)


def _safe_hash(path: Any) -> str:
    try:
        return sha256_file(path) if path and Path(path).exists() else ""
    except Exception:
        return ""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build ModelFin_111 package manifest")
    p.add_argument("--model_dir", "--model111_dir", dest="model_dir", default="ModelFin_111")
    p.add_argument("--state_model_dir", default="ModelFin_107A")
    p.add_argument("--state_eval_dir", default="EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only")
    p.add_argument("--split_manifest_json", default="Data/assb111/split_manifest.json")
    p.add_argument("--dataset_csv", default="Data/assb111/dataset_strict30.csv")
    p.add_argument("--features_csv", default="Data/assb111/features_107A_cycle.csv")
    p.add_argument("--input_file", default="input_assb111_strict30_saturating")
    p.add_argument("--leakage_audit_json", default="ModelFin_111/leakage_audit.json")
    p.add_argument("--feature_schema_json", default="", help="accepted for run-script compatibility; package uses ModelFin_111/feature_schema.json")
    p.add_argument("--feature_scaler_json", default="", help="accepted for run-script compatibility; package uses ModelFin_111/feature_scaler.json")
    p.add_argument("--overdecay_diagnostic_json", default="")
    p.add_argument("--training_summary_json", default="")
    p.add_argument("--checkpoint_manifest_csv", default="")
    p.add_argument("--selection_manifest_json", default="")
    p.add_argument("--protocol_audit_json", default="")
    p.add_argument("--selected_model_pt", default="")
    p.add_argument("--copy_state_engine", action="store_true", help="accepted for compatibility; ModelFin_111 uses reference-only state engine by default")
    args = p.parse_args(argv)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in [model_dir / "soh_head.pt", model_dir / "soh_head_config.json"] if not p.exists()]
    if missing:
        raise FileNotFoundError("ModelFin_111 SOH head is incomplete; missing " + ", ".join(missing))
    state_model_dir = Path(args.state_model_dir)
    state_eval_dir = Path(args.state_eval_dir)
    if not state_model_dir.exists():
        raise FileNotFoundError(f"state_model_dir not found: {state_model_dir}")
    if not state_eval_dir.exists():
        raise FileNotFoundError(f"state_eval_dir not found: {state_eval_dir}")
    manifest = load_manifest(args.split_manifest_json)

    for src, dst_name in [
        (args.split_manifest_json, "split_manifest.json"),
        (args.dataset_csv, "dataset.csv"),
        (args.features_csv, "features_107A_cycle.csv"),
        (args.leakage_audit_json, "leakage_audit.json"),
        (args.feature_schema_json, "feature_schema.json"),
        (args.feature_scaler_json, "feature_scaler.json"),
        (args.overdecay_diagnostic_json, "soh_overdecay_diagnostic.json"),
        (args.training_summary_json, "train_summary.json"),
        (args.checkpoint_manifest_csv, "checkpoint_manifest.csv"),
        (args.selection_manifest_json, "selection_manifest.json"),
        (args.protocol_audit_json, "protocol_audit.json"),
        (args.selected_model_pt, "selected_model.pt"),
    ]:
        if src and Path(src).exists():
            src_path = Path(src)
            dst_path = model_dir / dst_name
            try:
                same_file = src_path.resolve() == dst_path.resolve()
            except Exception:
                same_file = False
            if not same_file:
                shutil.copy2(src_path, dst_path)

    audit = _read_json(Path(args.leakage_audit_json)) or _read_json(model_dir / "leakage_audit.json")
    if audit and not bool(audit.get("ok", False)):
        raise RuntimeError("Refusing to package ModelFin_111 because leakage audit is not ok")
    cfg_payload = _read_json(model_dir / "soh_head_config.json")
    cfg = cfg_payload.get("config", {}) if isinstance(cfg_payload, dict) else {}
    train_summary = _read_json(Path(args.training_summary_json)) or _read_json(model_dir / "train_summary.json") or _read_json(model_dir / "training_summary.json")
    protocol_audit = _read_json(Path(args.protocol_audit_json)) or _read_json(model_dir / "protocol_audit.json") or _read_json(model_dir / "seed42_locked_protocol_audit.json")
    selection_manifest = _read_json(Path(args.selection_manifest_json)) or _read_json(model_dir / "selection_manifest.json")

    if not isinstance(train_summary, dict) or not train_summary:
        raise RuntimeError("Refusing to package ModelFin_111: missing train_summary.json/training_summary.json")
    status = str(train_summary.get("best_selection_status", "")).strip().lower()
    allowed_status = {"visible_guarded", "ema_visible_guarded", "topk_average_visible_guarded"}
    if status not in allowed_status:
        raise RuntimeError(f"Refusing to package ModelFin_111: unguarded or failed checkpoint status {status!r}")
    if bool(train_summary.get("test_metrics_used_for_selection", False)):
        raise RuntimeError("Refusing to package ModelFin_111: training summary says test metrics were used for selection")
    if not bool(train_summary.get("no_test_metrics_in_training_history", True)):
        raise RuntimeError("Refusing to package ModelFin_111: training history is not marked test-free")

    selected_path = Path(args.selected_model_pt) if str(args.selected_model_pt).strip() else (model_dir / "selected_model.pt")
    if selected_path.exists():
        selected_audit = _read_json(model_dir / "selected_checkpoint_audit.json") or _read_json(selected_path.with_suffix(".audit.json"))
        if not selected_audit or not bool(selected_audit.get("ok", False)):
            raise RuntimeError("Refusing to package selected_model.pt without an ok selected_checkpoint_audit.json")
        sel_status = str(selected_audit.get("best_selection_status", "")).strip().lower()
        if sel_status not in allowed_status:
            raise RuntimeError(f"Refusing to package selected_model.pt with non-guarded status {sel_status!r}")

    state_ref = {
        "state_engine_mode": "frozen_ModelFin_107A_reference",
        "state_model_dir": str(state_model_dir),
        "state_eval_dir": str(state_eval_dir),
        "state_model_hashes": _dir_file_hashes(state_model_dir),
        "state_eval_hashes": _dir_file_hashes(state_eval_dir),
        "note": "ModelFin_111 does not retrain or overwrite ModelFin_107A. Four states are evaluated from the frozen 107A reference outputs.",
    }
    _write_json(model_dir / "state_engine_ref.json", state_ref)

    package_files = {
        "soh_head_pt": _safe_hash(model_dir / "soh_head.pt"),
        "soh_head_config_json": _safe_hash(model_dir / "soh_head_config.json"),
        "feature_scaler_json": _safe_hash(model_dir / "feature_scaler.json"),
        "feature_schema_json": _safe_hash(model_dir / "feature_schema.json"),
        "train_history_csv": _safe_hash(model_dir / "train_history.csv"),
        "leakage_audit_json": _safe_hash(args.leakage_audit_json),
        "dataset_csv": _safe_hash(args.dataset_csv),
        "features_csv": _safe_hash(args.features_csv),
        "split_manifest_json": _safe_hash(args.split_manifest_json),
        "input_file": _safe_hash(args.input_file),
        "soh_overdecay_diagnostic_json": _safe_hash(args.overdecay_diagnostic_json),
        "training_summary_json": _safe_hash(args.training_summary_json or (model_dir / "train_summary.json")),
        "checkpoint_manifest_csv": _safe_hash(args.checkpoint_manifest_csv or (model_dir / "checkpoint_manifest.csv")),
        "selection_manifest_json": _safe_hash(args.selection_manifest_json or (model_dir / "selection_manifest.json")),
        "protocol_audit_json": _safe_hash(args.protocol_audit_json or (model_dir / "protocol_audit.json")),
        "selected_model_pt": _safe_hash(args.selected_model_pt or (model_dir / "selected_model.pt")),
        "selected_checkpoint_audit_json": _safe_hash(model_dir / "selected_checkpoint_audit.json"),
    }
    package = {
        "model_id": 111,
        "model_name": "ModelFin_111_seed42_locked_or_strict30_107A_states_plus_SOH_head",
        "candidate_tag": train_summary.get("candidate_tag", "") if isinstance(train_summary, dict) else "",
        "protocol_tag": train_summary.get("protocol_tag", "") if isinstance(train_summary, dict) else "",
        "seed": train_summary.get("seed", None) if isinstance(train_summary, dict) else None,
        "seed_locked": train_summary.get("seed_locked", False) if isinstance(train_summary, dict) else False,
        "soh_model_variant": cfg.get("model_variant", "unknown"),
        "soh_saturating_parameters": {k: cfg.get(k) for k in ["floor_min", "floor_max", "soh_floor_prior", "damage_rate_scale", "gate_gamma", "residual_bound", "soh_numeric_min", "w_floor_prior", "w_tail_guard"] if k in cfg},
        "strict_protocol": "SOH supervised loss uses train cycles only; test cycles 160-521 are held out.",
        "split_manifest": manifest,
        "state_engine_ref_json": "state_engine_ref.json",
        "state_guard_required": True,
        "package_files_sha256": package_files,
        "soh_head_config": cfg_payload,
        "leakage_audit_summary": audit,
        "training_summary": train_summary,
        "selection_manifest": selection_manifest,
        "protocol_audit_summary": protocol_audit,
        "selected_checkpoint_audit": _read_json(model_dir / "selected_checkpoint_audit.json"),
        "diagnostics": {
            "soh_overdecay_diagnostic_json": "soh_overdecay_diagnostic.json" if (model_dir / "soh_overdecay_diagnostic.json").exists() else "",
            "checkpoint_manifest_csv": "checkpoint_manifest.csv" if (model_dir / "checkpoint_manifest.csv").exists() else "",
            "train_summary_json": "train_summary.json" if (model_dir / "train_summary.json").exists() else "",
            "selected_model_pt": "selected_model.pt" if (model_dir / "selected_model.pt").exists() else "",
        },
        "package_sha256_without_self": "",
    }
    package["package_sha256_without_self"] = sha256_jsonable({k: v for k, v in package.items() if k != "package_sha256_without_self"})
    _write_json(model_dir / "model_manifest.json", package)
    _write_json(model_dir / "model111_manifest.json", package)
    print("[build_ModelFin111_package] wrote", model_dir / "model_manifest.json", "and", model_dir / "model111_manifest.json")
    print(json.dumps(_json_clean({"model_dir": str(model_dir), "state_model_dir": str(state_model_dir), "state_eval_dir": str(state_eval_dir), "soh_model_variant": package["soh_model_variant"], "package_sha256_without_self": package["package_sha256_without_self"], "leakage_audit_ok": audit.get("ok", None) if audit else None}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
