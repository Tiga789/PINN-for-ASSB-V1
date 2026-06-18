from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(p, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return default


def json_dump(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        f = open(p, "r", encoding="utf-8-sig", newline="")
    except Exception:
        return []
    with f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                fields.append(str(k)); seen.add(str(k))
    if not fields:
        fields = ["empty"]
        rows = [{"empty": ""}]
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def sha256_file(path: str | Path, max_bytes: Optional[int] = None) -> Optional[str]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        remaining = max_bytes
        while True:
            if remaining is not None and remaining <= 0:
                break
            chunk_size = 1024 * 1024
            if remaining is not None:
                chunk_size = min(chunk_size, remaining)
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
            if remaining is not None:
                remaining -= len(b)
    return h.hexdigest()


def file_info(path: str | Path, label: str, required: bool = False, hash_large: bool = False) -> Dict[str, Any]:
    p = Path(path) if path else Path("")
    exists = bool(str(path)) and p.exists()
    size = p.stat().st_size if exists and p.is_file() else None
    should_hash = exists and p.is_file() and (hash_large or (size is not None and size < 256 * 1024 * 1024))
    return {
        "label": label,
        "path": str(p) if str(path) else "",
        "exists": bool(exists),
        "required": bool(required),
        "size_bytes": size,
        "sha256": sha256_file(p) if should_hash else None,
        "hash_skipped_reason": None if should_hash else ("missing" if not exists else "large_file_hash_disabled"),
    }


def flatten_g3(g3_summary: Mapping[str, Any], g3_scorecard: Mapping[str, Any]) -> Dict[str, Any]:
    src = dict(g3_summary or {})
    if not src:
        src = dict(g3_scorecard or {})
    agg = src.get("frozen_test_per_target_aggregate") if isinstance(src.get("frozen_test_per_target_aggregate"), Mapping) else {}
    out = {
        "status": src.get("status"),
        "promotion_status": src.get("promotion_status"),
        "g4_ready": src.get("g4_ready"),
        "g4_blockers": src.get("g4_blockers"),
        "frozen_test_profile_count": (src.get("dataset") or {}).get("frozen_test_profile_count") if isinstance(src.get("dataset"), Mapping) else src.get("frozen_test_profile_count"),
        "frozen_test_mean_r2": safe_float(agg.get("all_target_profile_r2_mean")),
        "frozen_test_min_r2": safe_float(agg.get("all_target_profile_r2_min")),
        "frozen_test_phie_min_r2": safe_float(agg.get("phie_r2_min")),
        "worst_frozen_test_target_profile": src.get("worst_frozen_test_target_profile"),
    }
    return out


def aggregate_metrics_csv(rows: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    if not rows:
        return {"available": False}
    r2s = [safe_float(r.get("r2")) for r in rows]
    r2s = [v for v in r2s if math.isfinite(v)]
    targets = sorted({str(r.get("target", "")) for r in rows if r.get("target")})
    out: Dict[str, Any] = {
        "available": True,
        "row_count": len(rows),
        "mean_r2": float(sum(r2s) / len(r2s)) if r2s else float("nan"),
        "min_r2": float(min(r2s)) if r2s else float("nan"),
        "target_count": len(targets),
        "targets": targets,
    }
    for t in targets:
        vals = [safe_float(r.get("r2")) for r in rows if str(r.get("target")) == t]
        vals = [v for v in vals if math.isfinite(v)]
        if vals:
            out[f"{t}_r2_mean"] = float(sum(vals) / len(vals))
            out[f"{t}_r2_min"] = float(min(vals))
    finite_rows = [r for r in rows if math.isfinite(safe_float(r.get("r2")))]
    if finite_rows:
        wr = min(finite_rows, key=lambda r: safe_float(r.get("r2")))
        out["worst_row"] = dict(wr)
    return out


def copy_small_artifact(src: str | Path, dst_dir: str | Path, label: str, max_copy_mb: float = 50.0) -> Dict[str, Any]:
    p = Path(src)
    out = {"label": label, "source": str(p), "copied": False, "copied_to": "", "reason": ""}
    if not p.exists() or not p.is_file():
        out["reason"] = "missing"
        return out
    if p.stat().st_size > max_copy_mb * 1024 * 1024:
        out["reason"] = f"larger_than_{max_copy_mb}MB"
        return out
    dst = Path(dst_dir) / p.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dst)
    out.update({"copied": True, "copied_to": str(dst), "reason": "copied"})
    return out


def try_torch_speed(checkpoint_path: str | Path, device_arg: str = "auto", trials: int = 200, batch_size: int = 8192, warmup: int = 10) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": "NOT_RUN",
        "reason": "",
        "checkpoint": str(checkpoint_path),
        "device": None,
        "batch_size": int(batch_size),
        "trials": int(trials),
    }
    p = Path(checkpoint_path) if checkpoint_path else Path("")
    if not p.exists():
        result.update({"status": "SKIPPED", "reason": "checkpoint_missing"})
        return result
    try:
        import numpy as np
        import torch
        try:
            from gv1.d17_g.g3_frozen_audit import torch_load_safe, build_model_from_checkpoint
        except Exception as e:
            result.update({"status": "SKIPPED", "reason": f"could_not_import_g3_model_builder: {e!r}"})
            return result
        if str(device_arg).lower() == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_arg)
        ckpt = torch_load_safe(p, map_location="cpu")
        model = build_model_from_checkpoint(ckpt, device)
        x_mean = np.asarray(ckpt.get("x_mean"), dtype=np.float32)
        if x_mean.size <= 0:
            raise ValueError("checkpoint x_mean is empty")
        x = torch.zeros((int(batch_size), int(x_mean.size)), dtype=torch.float32, device=device)
        model.eval()
        with torch.no_grad():
            for _ in range(max(0, int(warmup))):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(max(1, int(trials))):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        samples = int(batch_size) * max(1, int(trials))
        result.update({
            "status": "PASS",
            "reason": "synthetic_forward_only_speed; no soft-label inputs used",
            "device": str(device),
            "input_dim": int(x_mean.size),
            "output_dim": int(np.asarray(ckpt.get("y_mean")).size) if ckpt.get("y_mean") is not None else None,
            "elapsed_s": float(elapsed),
            "samples": int(samples),
            "samples_per_second": float(samples / elapsed) if elapsed > 0 else float("inf"),
            "latency_ms_per_batch": float(1000.0 * elapsed / max(1, int(trials))),
            "latency_us_per_sample": float(1e6 * elapsed / max(1, samples)),
        })
        return result
    except Exception as e:
        result.update({"status": "REVIEW", "reason": repr(e)})
        return result


def render_markdown(scorecard: Mapping[str, Any]) -> str:
    g3 = scorecard.get("g3") or {}
    speed = scorecard.get("speed_audit") or {}
    lines = []
    lines.append("# D17-G4 Final Scorecard and Speed Audit")
    lines.append("")
    lines.append(f"Created: `{scorecard.get('created_at_utc')}`")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- `status`: **{scorecard.get('status')}**")
    lines.append(f"- `final_candidate_ready`: **{scorecard.get('final_candidate_ready')}**")
    lines.append(f"- `recommendation`: `{scorecard.get('recommendation')}`")
    blockers = scorecard.get("blockers") or []
    if blockers:
        lines.append("- blockers:")
        for b in blockers:
            lines.append(f"  - {b}")
    lines.append("")
    lines.append("## Frozen-test report-only state audit")
    lines.append("")
    lines.append(f"- G3 status / promotion: `{g3.get('status')}` / `{g3.get('promotion_status')}`")
    lines.append(f"- Frozen-test mean R²: `{g3.get('frozen_test_mean_r2')}`")
    lines.append(f"- Frozen-test min R²: `{g3.get('frozen_test_min_r2')}`")
    lines.append(f"- Frozen-test phie min R²: `{g3.get('frozen_test_phie_min_r2')}`")
    lines.append("")
    lines.append("## Policy boundary")
    lines.append("")
    lines.append("- G2.1 used train-cell soft labels for supervised generator-surrogate training.")
    lines.append("- Validation soft labels were report-only in G2.1.")
    lines.append("- G3 used frozen-test soft labels only once for report-only audit.")
    lines.append("- G4 performs no training and no checkpoint selection.")
    lines.append("")
    lines.append("## Speed audit")
    lines.append("")
    lines.append(f"- Speed status: `{speed.get('status')}`")
    lines.append(f"- Device: `{speed.get('device')}`")
    lines.append(f"- Samples/s: `{speed.get('samples_per_second')}`")
    lines.append(f"- Latency per batch ms: `{speed.get('latency_ms_per_batch')}`")
    lines.append(f"- Note: `{speed.get('reason')}`")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for k, v in (scorecard.get("files") or {}).items():
        lines.append(f"- `{k}`: `{v}`")
    return "\n".join(lines) + "\n"


def run_g4_finalization(
    *,
    config: Mapping[str, Any],
    g0_audit: str | Path,
    g0_profile_semantics_csv: str | Path,
    g21_summary: str | Path,
    g21_dir: str | Path,
    g3_summary: str | Path,
    g3_scorecard: str | Path,
    g3_dir: str | Path,
    out_dir: str | Path,
    checkpoint: str | Path = "",
    device: str = "auto",
    speed_trials: int = 200,
    speed_batch_size: int = 8192,
    hash_large_artifacts: bool = False,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_dir = out / "final_report_artifacts"
    report_dir.mkdir(parents=True, exist_ok=True)

    g0 = read_json(g0_audit, default={}) or {}
    g21 = read_json(g21_summary, default={}) or {}
    g3s = read_json(g3_summary, default={}) or {}
    g3c = read_json(g3_scorecard, default={}) or {}
    g3 = flatten_g3(g3s, g3c)

    checkpoint_path = str(checkpoint or "")
    if not checkpoint_path:
        files = g21.get("files") if isinstance(g21.get("files"), Mapping) else {}
        checkpoint_path = str(files.get("best_model_pt") or Path(g21_dir) / "model" / "best_model.pt")

    # Inputs and artifact manifests.
    critical = [
        (g0_audit, "G0 generator equivalence audit", True),
        (g0_profile_semantics_csv, "G0 profile semantics CSV", True),
        (g21_summary, "G2.1 candidate summary", True),
        (checkpoint_path, "G2.1 best model checkpoint", True),
        (g3_summary, "G3 frozen-test summary", False),
        (g3_scorecard, "G3 scorecard", True),
        (Path(g3_dir) / "D17_G3_PER_TARGET_PROFILE_METRICS.csv", "G3 per-target profile metrics", False),
        (Path(g3_dir) / "D17_G3_PREDICTION_MANIFEST.csv", "G3 prediction manifest", False),
        (Path(g3_dir) / "D17_G3_PHIE_AUDIT.csv", "G3 phie audit", False),
    ]
    artifact_rows = [file_info(p, label, required=req, hash_large=bool(hash_large_artifacts)) for p, label, req in critical]
    write_csv(artifact_rows, out / "D17_G4_ARTIFACT_MANIFEST.csv")

    copied = []
    for p, label, _ in critical:
        copied.append(copy_small_artifact(p, report_dir, label, max_copy_mb=float(config.get("max_copy_artifact_mb", 50.0))))
    write_csv(copied, out / "D17_G4_COPIED_ARTIFACTS.csv")

    metrics_rows = read_csv(Path(g3_dir) / "D17_G3_PER_TARGET_PROFILE_METRICS.csv")
    metrics_agg = aggregate_metrics_csv(metrics_rows)

    speed = try_torch_speed(checkpoint_path, device_arg=device, trials=int(speed_trials), batch_size=int(speed_batch_size), warmup=int(config.get("speed_warmup", 10)))
    json_dump(speed, out / "D17_G4_SPEED_AUDIT.json")

    blockers: List[str] = []
    if str(g0.get("status")) != "PASS":
        blockers.append(f"G0 audit not PASS: {g0.get('status')}")
    if str(g21.get("status")) != "PASS" or not bool(g21.get("g3_ready")):
        blockers.append(f"G2.1 candidate not PASS/g3_ready=true: status={g21.get('status')} g3_ready={g21.get('g3_ready')}")
    if str(g3.get("status")) != "PASS" or str(g3.get("promotion_status")) != "PASS" or not bool(g3.get("g4_ready")):
        blockers.append(f"G3 frozen audit not PASS/promotion PASS/g4_ready=true: {g3}")
    for row in artifact_rows:
        if row.get("required") and not row.get("exists"):
            blockers.append(f"required artifact missing: {row.get('label')} -> {row.get('path')}")

    mean_thr = float(config.get("frozen_mean_r2_threshold", 0.95))
    min_thr = float(config.get("frozen_min_r2_threshold", 0.90))
    phie_thr = float(config.get("frozen_phie_min_r2_threshold", 0.90))
    if math.isfinite(g3.get("frozen_test_mean_r2", float("nan"))) and g3["frozen_test_mean_r2"] < mean_thr:
        blockers.append(f"frozen mean R2 below {mean_thr}: {g3['frozen_test_mean_r2']}")
    if math.isfinite(g3.get("frozen_test_min_r2", float("nan"))) and g3["frozen_test_min_r2"] < min_thr:
        blockers.append(f"frozen min R2 below {min_thr}: {g3['frozen_test_min_r2']}")
    if math.isfinite(g3.get("frozen_test_phie_min_r2", float("nan"))) and g3["frozen_test_phie_min_r2"] < phie_thr:
        blockers.append(f"frozen phie min R2 below {phie_thr}: {g3['frozen_test_phie_min_r2']}")

    # Speed is advisory by default. A missing speed audit should not invalidate a
    # scientific scorecard; it is marked REVIEW in the speed JSON.
    require_speed = bool(config.get("require_speed_pass", False))
    if require_speed and str(speed.get("status")) != "PASS":
        blockers.append(f"speed audit required but not PASS: {speed.get('status')} {speed.get('reason')}")

    status = "PASS" if not blockers else "REVIEW"
    final_candidate_ready = bool(status == "PASS")
    recommendation = "FREEZE_D17_G_GENERATOR_SURROGATE_AS_G4_CANDIDATE" if final_candidate_ready else "DO_NOT_FREEZE_REVIEW_G4_BLOCKERS"
    scorecard: Dict[str, Any] = {
        "protocol": "D17-G4_FINAL_SCORECARD_SPEED_AUDIT",
        "created_at_utc": utc_now(),
        "status": status,
        "final_candidate_ready": final_candidate_ready,
        "recommendation": recommendation,
        "blockers": blockers,
        "purpose": "Final scorecard, freeze manifest, artifact audit, and optional model-forward speed audit after G3 frozen-test report-only PASS.",
        "policy": {
            "training_performed_in_G4": False,
            "checkpoint_selection_performed_in_G4": False,
            "split_modified_in_G4": False,
            "frozen_test_feedback_used_for_training": False,
            "train_cell_softlabels_used_upstream_G2_1": True,
            "validation_softlabels_report_only_upstream_G2_1": True,
            "frozen_test_softlabels_report_only_upstream_G3": True,
            "G4_is_report_and_export_only": True,
        },
        "g0": {
            "status": g0.get("status"),
            "g1_ready": g0.get("g1_ready"),
            "profile_count_audited": g0.get("profile_count_audited"),
            "semantics_known_fraction": g0.get("semantics_known_fraction"),
        },
        "g21": {
            "status": g21.get("status"),
            "g3_ready": g21.get("g3_ready"),
            "recommendation": g21.get("recommendation"),
            "best_epoch": g21.get("best_epoch"),
            "dataset": g21.get("dataset"),
            "model": g21.get("model"),
            "policy": g21.get("policy"),
            "fit_train_per_target_aggregate": g21.get("fit_train_per_target_aggregate"),
            "internal_heldout_per_target_aggregate": g21.get("internal_heldout_per_target_aggregate"),
            "validation_report_only_per_target_aggregate": g21.get("validation_report_only_per_target_aggregate"),
        },
        "g3": g3,
        "g3_metrics_csv_aggregate": metrics_agg,
        "speed_audit": speed,
        "artifact_manifest": artifact_rows,
        "files": {
            "final_scorecard_json": str(out / "D17_G4_FINAL_SCORECARD.json"),
            "final_report_md": str(out / "D17_G4_FINAL_REPORT.md"),
            "speed_audit_json": str(out / "D17_G4_SPEED_AUDIT.json"),
            "artifact_manifest_csv": str(out / "D17_G4_ARTIFACT_MANIFEST.csv"),
            "copied_artifacts_csv": str(out / "D17_G4_COPIED_ARTIFACTS.csv"),
            "frozen_candidate_manifest_json": str(out / "D17_G4_FROZEN_CANDIDATE_MANIFEST.json"),
        },
    }
    frozen_manifest = {
        "protocol": "D17-G4_FROZEN_CANDIDATE_MANIFEST",
        "created_at_utc": scorecard["created_at_utc"],
        "candidate_id": "D17-G2.1/G3/G4_generator_surrogate_candidate",
        "final_candidate_ready": final_candidate_ready,
        "checkpoint": checkpoint_path,
        "g0_audit": str(g0_audit),
        "g0_profile_semantics_csv": str(g0_profile_semantics_csv),
        "g21_summary": str(g21_summary),
        "g3_summary": str(g3_summary),
        "g3_scorecard": str(g3_scorecard),
        "artifact_manifest_csv": str(out / "D17_G4_ARTIFACT_MANIFEST.csv"),
        "policy": scorecard["policy"],
    }
    json_dump(frozen_manifest, out / "D17_G4_FROZEN_CANDIDATE_MANIFEST.json")
    json_dump(scorecard, out / "D17_G4_FINAL_SCORECARD.json")
    (out / "D17_G4_FINAL_REPORT.md").write_text(render_markdown(scorecard), encoding="utf-8")
    return scorecard
