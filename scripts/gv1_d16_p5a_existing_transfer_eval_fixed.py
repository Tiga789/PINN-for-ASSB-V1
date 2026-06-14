from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_nn.data import (
    build_features,
    build_targets,
    discover_npz,
    load_profile_arrays,
    profile_id_from_path,
)
from gv1.p2dlite_rg_nn.metrics import compute_rg_metrics, thresholds_status
from gv1.p2dlite_rg_nn.model import build_model
from gv1.p2dlite_rg_nn.train_eval import predict_numpy
from gv1.p2dlite_rg_nn.utils import load_json, write_csv, write_json

try:
    from gv1.p2dlite_rg_boundary.projection import apply_theta_projection
except Exception:  # pragma: no cover - fallback for older trees
    apply_theta_projection = None

try:
    from gv1.p2dlite_rg_nn_precision.audit import (
        aggregate_rows,
        audit_prediction_file,
        precision_status,
    )
except Exception:  # pragma: no cover - still allow inference/eval without precision audit
    aggregate_rows = None
    audit_prediction_file = None
    precision_status = None


def _json_default(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if not math.isfinite(v) else v
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return str(x)


def _write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _write_csv(rows: List[Mapping[str, Any]], path: Path) -> None:
    # Prefer project helper if available, but keep this script self-contained.
    path.parent.mkdir(parents=True, exist_ok=True)
    if write_csv is not None:
        write_csv(rows, path)
        return
    import csv
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _device(name: str):
    import torch

    if name in {"", "auto", None}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(name))


def _safe_name(s: str) -> str:
    return str(s).replace("/", "__").replace("\\", "__").replace(":", "_").replace(" ", "_")


def _checkpoint_paths_for_model_dir(model_dir: Path) -> List[Path]:
    return [
        model_dir / "model" / "best_with_state.pt",
        model_dir / "best_with_state.pt",
        model_dir / "checkpoint" / "best_with_state.pt",
        model_dir / "checkpoints" / "best_with_state.pt",
        model_dir / "model" / "best.pt",
        model_dir / "best.pt",
        model_dir / "checkpoint" / "best.pt",
        model_dir / "checkpoints" / "best.pt",
    ]


def _model_dir_from_checkpoint_file(p: Path) -> Path:
    # If checkpoint is inside a model/checkpoint/checkpoints subfolder, return the run root.
    if p.parent.name.lower() in {"model", "checkpoint", "checkpoints"}:
        return p.parent.parent
    return p.parent


def _model_file(model_dir: Path) -> Path:
    candidates = _checkpoint_paths_for_model_dir(model_dir)
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find model checkpoint. Tried:\n  " + "\n  ".join(str(x) for x in candidates)
    )


def _looks_like_model_dir_name(name: str) -> bool:
    s = name.lower()
    positive = ["d15", "p2", "p1", "p3", "p5b", "p5c", "precision", "closedset", "nn", "benchmark", "model"]
    negative = [
        "softlabel", "softlabels", "replay", "profile", "profiles", "manifest", "audit", "standard",
        "summary", "plot", "plots", "review", "zip", "all55_final", "batch134", "batch56", "cache_staging",
    ]
    if any(x in s for x in negative):
        return False
    return any(x in s for x in positive)


def _score_model_candidate(model_dir: Path, ckpt: Path) -> int:
    s = (str(model_dir) + " " + str(ckpt)).lower()
    score = 0
    if "d15" in s: score += 100
    if "p2" in s: score += 80
    if "precision" in s: score += 50
    if "benchmark" in s: score += 40
    if "p1" in s: score += 30
    if "p3" in s: score += 25
    if "p5b" in s or "p5c" in s: score += 15
    if "best_with_state.pt" in s: score += 10
    if "softlabel" in s or "replay" in s or "audit" in s: score -= 200
    return score


def _dedupe_candidates(cands: List[Tuple[Path, Path, str]]) -> List[Tuple[Path, Path, str]]:
    seen = set()
    out = []
    for md, ck, reason in cands:
        key = str(md.resolve()).lower() if md.exists() else str(md).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((md, ck, reason))
    return out


def _discover_model_candidates(cache_root: Optional[Path], project_root: Path, explicit_model_dir: Optional[Path] = None) -> List[Tuple[Path, Path, str]]:
    cands: List[Tuple[Path, Path, str]] = []

    def add_dir(d: Path, reason: str) -> None:
        try:
            for ck in _checkpoint_paths_for_model_dir(d):
                if ck.exists():
                    cands.append((d, ck, reason))
                    return
        except Exception:
            return

    if explicit_model_dir is not None and str(explicit_model_dir).strip().lower() not in {"", "auto"}:
        add_dir(explicit_model_dir, "explicit_model_dir")

    # Fast first-level scan: most project run dirs are direct children of the cache root or project root.
    roots = []
    if cache_root is not None:
        roots.append(cache_root)
    roots.append(project_root)
    for root in roots:
        if not root or not root.exists():
            continue
        add_dir(root, f"root_direct:{root}")
        try:
            for child in root.iterdir():
                if child.is_dir():
                    add_dir(child, f"first_level:{root}")
                    if _looks_like_model_dir_name(child.name):
                        # One extra level for run_dir/model/checkpoint variants and nested wrappers.
                        try:
                            for g in child.iterdir():
                                if g.is_dir():
                                    add_dir(g, f"second_level:{child}")
                        except Exception:
                            pass
        except Exception:
            pass

    # Targeted recursive scan only under likely folders; skip huge soft-label/profile trees.
    for root in roots:
        if not root or not root.exists():
            continue
        try:
            for child in root.iterdir():
                if not child.is_dir() or not _looks_like_model_dir_name(child.name):
                    continue
                for pattern in ("best_with_state.pt", "best.pt"):
                    try:
                        for ck in child.rglob(pattern):
                            # Avoid pulling checkpoints from audit/review copies unless no better candidate exists.
                            if any(part.lower() in {"__pycache__"} for part in ck.parts):
                                continue
                            md = _model_dir_from_checkpoint_file(ck)
                            cands.append((md, ck, f"targeted_recursive:{child.name}"))
                    except Exception:
                        pass
        except Exception:
            pass

    cands = _dedupe_candidates(cands)
    cands.sort(key=lambda x: _score_model_candidate(x[0], x[1]), reverse=True)
    return cands


def _resolve_and_load_checkpoint(args: argparse.Namespace, device, project_root: Path) -> Tuple[Path, Path, Mapping[str, Any], Dict[str, Any]]:
    softlabel_dir = Path(args.softlabel_dir)
    cache_root = Path(args.cache_root) if args.cache_root else softlabel_dir.parent
    explicit = Path(args.model_dir) if args.model_dir and str(args.model_dir).lower() != "auto" else None
    candidates = _discover_model_candidates(cache_root, project_root, explicit_model_dir=explicit)
    discovery: Dict[str, Any] = {
        "requested_model_dir": str(args.model_dir),
        "cache_root": str(cache_root),
        "candidate_count": len(candidates),
        "candidates": [
            {"rank": i + 1, "model_dir": str(md), "checkpoint": str(ck), "reason": reason, "score": _score_model_candidate(md, ck)}
            for i, (md, ck, reason) in enumerate(candidates[:50])
        ],
    }
    load_errors: List[Dict[str, Any]] = []
    import torch
    for md, ckpt, reason in candidates:
        try:
            ck = torch.load(ckpt, map_location=device, weights_only=False)
            if not isinstance(ck, Mapping) or "state" not in ck:
                raise KeyError("checkpoint has no top-level 'state' key")
            state = ck["state"]
            required = ["target_slices", "profile_ids", "input_dim", "output_dim", "model_config", "model_state_dict"]
            missing = [k for k in required if k not in state and k != "model_state_dict"]
            # model_state_dict is top-level in D15 checkpoints, not inside state.
            if "model_state_dict" not in ck:
                missing.append("model_state_dict")
            if missing:
                raise KeyError(f"checkpoint missing required D15 RG NN keys: {missing}")
            discovery["selected"] = {"model_dir": str(md), "checkpoint": str(ckpt), "reason": reason, "score": _score_model_candidate(md, ckpt)}
            discovery["load_errors"] = load_errors[:20]
            return md, ckpt, ck, discovery
        except Exception as exc:
            load_errors.append({"model_dir": str(md), "checkpoint": str(ckpt), "error": repr(exc)})
            continue
    discovery["load_errors"] = load_errors[:50]
    # Write a readable message to stderr before raising. The run wrapper also writes discovery JSON if possible.
    msg = [
        "No compatible existing D15 RG NN checkpoint could be auto-discovered.",
        f"Requested model_dir={args.model_dir}",
        f"Cache root searched={cache_root}",
        "Look for a directory containing model/best_with_state.pt, best_with_state.pt, model/best.pt, or best.pt.",
        "Top discovered candidates:",
    ]
    for c in discovery["candidates"][:10]:
        msg.append(f"  rank={c['rank']} score={c['score']} model_dir={c['model_dir']} checkpoint={c['checkpoint']} reason={c['reason']}")
    if load_errors:
        msg.append("Load errors for top candidates:")
        for e in load_errors[:10]:
            msg.append(f"  {e['checkpoint']}: {e['error']}")
    raise FileNotFoundError("\n".join(msg))


def _scalar_string(x: Any, default: str = "") -> str:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return default
        val = arr.reshape(-1)[0]
        if isinstance(val, bytes):
            return val.decode("utf-8", errors="ignore")
        return str(val)
    except Exception:
        return default


def _extract_battery_num(text: str) -> Optional[int]:
    m = re.search(r"battery[-_ ]?(\d+)", text, flags=re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"cell[-_ ]?(\d+)", text, flags=re.I)
    if m:
        return int(m.group(1))
    return None


def _extract_batch(text: str) -> str:
    m = re.search(r"Batch[-_ ]?(\d+)", text, flags=re.I)
    return f"Batch-{int(m.group(1))}" if m else ""


def _extract_protocol(text: str) -> str:
    # Keep common XJTU tags stable. 3C is used for Batch-2, R2.5/R3 for Batch-3/4, random/GEO for Batch-5/6.
    s = text.lower()
    if "r2.5" in s or "r2_5" in s or "r25" in s:
        return "R2.5"
    if re.search(r"\br3\b", s) or "_r3_" in s or "-r3-" in s:
        return "R3"
    if "3c" in s:
        return "3C"
    if "2c" in s:
        return "2C"
    if "random" in s or "walk" in s:
        return "random_walk"
    if "geo" in s:
        return "GEO"
    return ""


def _profile_meta_from_id(pid: str) -> Dict[str, Any]:
    return {
        "profile_id": pid,
        "batch": _extract_batch(pid),
        "protocol": _extract_protocol(pid),
        "battery_num": _extract_battery_num(pid),
    }


def _profile_meta_from_npz(npz_path: Path, root: Path) -> Dict[str, Any]:
    pid = profile_id_from_path(npz_path, root)
    meta = _profile_meta_from_id(pid)
    try:
        with np.load(npz_path, allow_pickle=True) as z:
            raw_batch = _scalar_string(z["batch"], "") if "batch" in z.files else ""
            raw_protocol = _scalar_string(z["protocol"], "") if "protocol" in z.files else ""
            raw_cell = _scalar_string(z["cell_uid"], "") if "cell_uid" in z.files else ""
        if raw_batch:
            meta["batch"] = _extract_batch(raw_batch) or raw_batch
        if raw_protocol:
            meta["protocol"] = _extract_protocol(raw_protocol) or raw_protocol
        if raw_cell and meta.get("battery_num") is None:
            meta["battery_num"] = _extract_battery_num(raw_cell)
        meta["cell_uid_raw"] = raw_cell
    except Exception as exc:
        meta["meta_read_warning"] = repr(exc)
    return meta


def _build_seen_meta(profile_ids: Sequence[str]) -> List[Dict[str, Any]]:
    out = []
    for i, pid in enumerate(profile_ids):
        m = _profile_meta_from_id(pid)
        m["seen_index"] = i
        out.append(m)
    return out


def _nearest_by_battery(candidates: List[Dict[str, Any]], target_batt: Optional[int]) -> Dict[str, Any]:
    if not candidates:
        raise ValueError("empty candidates")
    if target_batt is None:
        return candidates[0]
    return sorted(
        candidates,
        key=lambda c: abs((c.get("battery_num") if c.get("battery_num") is not None else target_batt) - target_batt),
    )[0]


def route_to_seen_profile(target: Dict[str, Any], seen: List[Dict[str, Any]]) -> Tuple[int, str, str]:
    """Map an ALL55 profile to an in-range D15-P2 one-hot profile index.

    D15-P2/P1 models were trained with an 8-cell profile one-hot input. For an unseen ALL55 profile,
    using its default sorted index can exceed the one-hot width and fail before predictions are written.
    This router chooses a proxy profile index from the model's seen profile_ids.
    """
    pid = target.get("profile_id", "")
    for s in seen:
        if s.get("profile_id") == pid:
            return int(s["seen_index"]), str(s["profile_id"]), "exact_profile_id"

    tb = target.get("batch", "")
    tp = target.get("protocol", "")
    tbn = target.get("battery_num")

    same_batch_protocol = [s for s in seen if s.get("batch") == tb and s.get("protocol") == tp and tb and tp]
    if same_batch_protocol:
        s = _nearest_by_battery(same_batch_protocol, tbn)
        return int(s["seen_index"]), str(s["profile_id"]), "same_batch_protocol_nearest_battery"

    same_protocol = [s for s in seen if s.get("protocol") == tp and tp]
    if same_protocol:
        s = _nearest_by_battery(same_protocol, tbn)
        return int(s["seen_index"]), str(s["profile_id"]), "same_protocol_nearest_battery"

    same_batch = [s for s in seen if s.get("batch") == tb and tb]
    if same_batch:
        s = _nearest_by_battery(same_batch, tbn)
        return int(s["seen_index"]), str(s["profile_id"]), "same_batch_nearest_battery"

    # Batch-2 has no D15-P2 seen profile. Route to Batch-1/2C fixed full-cycle if available.
    if tb == "Batch-2":
        b1 = [s for s in seen if s.get("batch") == "Batch-1"]
        if b1:
            s = _nearest_by_battery(b1, tbn)
            return int(s["seen_index"]), str(s["profile_id"]), "batch2_fallback_to_batch1_fixed_fullcycle"

    s = seen[0]
    return int(s["seen_index"]), str(s["profile_id"]), "fallback_first_seen_profile"


def _normalize_slices(slices: Mapping[str, Any]) -> Dict[str, Tuple[int, int]]:
    return {str(k): (int(v[0]), int(v[1])) for k, v in slices.items()}


def _apply_projection(y_pred: np.ndarray, target_slices: Mapping[str, Tuple[int, int]], theta_min: float, theta_max: float) -> np.ndarray:
    if apply_theta_projection is not None:
        return apply_theta_projection(y_pred, target_slices, theta_min=theta_min, theta_max=theta_max, apply_to=("theta_a", "theta_c"))
    yp = np.asarray(y_pred, dtype=np.float32).copy()
    for key in ("theta_a", "theta_c"):
        s, e = target_slices[key]
        yp[:, s:e] = np.clip(yp[:, s:e], theta_min, theta_max)
    return yp


def _prefix_metrics(metrics: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def _metrics_row(profile_id: str, mode: str, y_true: np.ndarray, y_pred: np.ndarray, slices: Mapping[str, Tuple[int, int]], extra: Mapping[str, Any]) -> Dict[str, Any]:
    m = compute_rg_metrics(y_true, y_pred, slices)
    row = dict(extra)
    row.update({"profile_id": profile_id, "projection_mode": mode})
    row.update(m)
    return row


def _aggregate_group(rows: List[Mapping[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for r in rows:
        k = tuple(r.get(x, "") for x in keys)
        groups.setdefault(k, []).append(r)
    out: List[Dict[str, Any]] = []
    metric_keys = [
        "phis_c_mae", "phie_mae", "theta_a_mae", "theta_c_mae",
        "theta_a_mean_mae", "theta_c_mean_mae",
        "grad_a_surface_center_mae", "grad_c_surface_center_mae",
        "pred_theta_outside_fraction", "pred_theta_boundary_hit_fraction",
        "min_selected_corr",
    ]
    for k, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        row: Dict[str, Any] = {keys[i]: k[i] for i in range(len(keys))}
        row["profile_count"] = len(rs)
        for mk in metric_keys:
            vals = []
            for r in rs:
                try:
                    v = float(r.get(mk, float("nan")))
                    if np.isfinite(v):
                        vals.append(v)
                except Exception:
                    pass
            if vals:
                row[f"{mk}_mean"] = float(np.mean(vals))
                row[f"{mk}_max"] = float(np.max(vals))
                row[f"{mk}_min"] = float(np.min(vals))
        out.append(row)
    return out


def _run_internal_precision_audit(eval_dir: Path, softlabel_dir: Path, audit_dir: Path, audit_cfg: Mapping[str, Any], filename: str) -> Dict[str, Any]:
    audit_dir.mkdir(parents=True, exist_ok=True)
    if audit_prediction_file is None or aggregate_rows is None or precision_status is None:
        summary = {
            "stage": "D16-P5A internal precision audit",
            "overall_status": "REVIEW",
            "reason": "gv1.p2dlite_rg_nn_precision.audit import failed; predictions were still generated.",
        }
        _write_json(summary, audit_dir / "D15_P2_PRECISION_AUDIT_SUMMARY.json")
        return summary

    pred_root = eval_dir / "predictions"
    preds = sorted(pred_root.rglob("*.npz"))
    rows: List[Dict[str, Any]] = []
    top_rows: List[Dict[str, Any]] = []
    cycle_rows: List[Dict[str, Any]] = []
    failures: List[str] = []

    for p in preds:
        try:
            row, top, cyc = audit_prediction_file(p, softlabel_dir, audit_cfg, filename=filename)
            rows.append(row)
            top_rows.extend(top)
            cycle_rows.extend(cyc)
        except Exception as exc:
            failures.append(f"{p}: {exc!r}")

    aggregate = aggregate_rows(rows)
    status = precision_status(rows, aggregate, audit_cfg)
    if failures:
        status["overall_status"] = "FAIL"
        status["read_failures"] = failures
    summary = {
        "stage": "D16-P5A internal compatibility precision audit using D15-P2 audit format",
        "softlabel_dir": str(softlabel_dir),
        "eval_dir": str(eval_dir),
        "prediction_file_count": len(preds),
        "profile_count_audited": len(rows),
        "aggregate": aggregate,
        "precision_status": status,
        "overall_status": status.get("overall_status", "FAIL"),
        "failures": failures,
    }
    _write_csv(rows, audit_dir / "D15_P2_PRECISION_AUDIT_BY_PROFILE.csv")
    _write_json(rows, audit_dir / "D15_P2_PRECISION_AUDIT_BY_PROFILE.json")
    _write_csv(top_rows, audit_dir / "D15_P2_TOPK_ERROR_WINDOWS.csv")
    _write_json(top_rows, audit_dir / "D15_P2_TOPK_ERROR_WINDOWS.json")
    _write_csv(cycle_rows, audit_dir / "D15_P2_CYCLE_LEVEL_AUDIT.csv")
    _write_json(summary, audit_dir / "D15_P2_PRECISION_AUDIT_SUMMARY.json")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="D16-P5A fixed ALL55 existing-model transfer evaluator. Generates predictions before audit and routes unseen ALL55 profiles into the D15-P2 model's 8-cell one-hot space."
    )
    p.add_argument("--softlabel-dir", required=True, help="D15 ALL55 root containing one folder per cell and solution_softlabels.npz")
    p.add_argument("--model-dir", default="auto", help="Existing D15-P2/P3/P3C model directory. Use 'auto' to discover under --cache-root / project root.")
    p.add_argument("--cache-root", default=None, help="GV1 cache root used for automatic model discovery. Default: parent of --softlabel-dir.")
    p.add_argument("--run-dir", required=True, help="Output root, e.g. E:/.../xjtu_d16_p5a_fixed_D15P2_existing_on_ALL55")
    p.add_argument("--config", default="configs/d15_p2_precision_benchmark_config.json")
    p.add_argument("--filename", default="solution_softlabels.npz")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=65536)
    p.add_argument("--eval-stride", type=int, default=None)
    p.add_argument("--allow-overwrite", action="store_true")
    p.add_argument("--primary-mode", choices=["raw", "projected"], default="projected", help="Which prediction is written to eval_full_profiles/predictions for D15-P2 precision_audit compatibility.")
    p.add_argument("--theta-min", type=float, default=1e-4)
    p.add_argument("--theta-max", type=float, default=0.9999)
    p.add_argument("--limit-cells", type=int, default=0, help="Optional smoke limit. 0 means all cells.")
    p.add_argument("--no-internal-audit", action="store_true", help="Only write predictions/metrics; skip internal precision audit.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    run_dir = Path(args.run_dir)
    eval_dir = run_dir / "eval_full_profiles"
    audit_dir = run_dir / "precision_audit"
    softlabel_dir = Path(args.softlabel_dir)
    model_dir = Path(args.model_dir)
    cfg = load_json(args.config)
    data_cfg = cfg.get("data", {})
    thresholds = cfg.get("scorecard_thresholds", {})
    audit_cfg = cfg.get("precision_audit", {})

    if run_dir.exists() and any(run_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f"run-dir exists and is non-empty: {run_dir}; pass --allow-overwrite to rerun")
    if run_dir.exists() and args.allow_overwrite:
        shutil.rmtree(run_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = _device(args.device)
    try:
        model_dir, mf, ck, model_discovery = _resolve_and_load_checkpoint(args, device, ROOT)
    except Exception as exc:
        # Best-effort discovery dump for failed runs.
        failure_dir = Path(args.run_dir)
        failure_dir.mkdir(parents=True, exist_ok=True)
        _write_json({"stage": "D16-P5A model discovery failure", "error": repr(exc), "requested_model_dir": str(args.model_dir), "softlabel_dir": str(args.softlabel_dir), "cache_root": str(args.cache_root or Path(args.softlabel_dir).parent)}, failure_dir / "D16_P5A_MODEL_DISCOVERY_FAILURE.json")
        raise
    _write_json(model_discovery, run_dir / "D16_P5A_MODEL_DISCOVERY.json")
    if "state" not in ck:
        raise KeyError(f"Checkpoint {mf} has no 'state' key; this fixed runner expects D15 RG NN checkpoint format.")
    state = ck["state"]
    target_slices = _normalize_slices(state["target_slices"])
    profile_ids = [str(x) for x in list(state.get("profile_ids", []))]
    if not profile_ids:
        raise ValueError("Model state has no profile_ids; cannot route ALL55 transfer profiles.")
    seen = _build_seen_meta(profile_ids)

    model = build_model(int(state["input_dim"]), int(state["output_dim"]), state["model_config"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    files = discover_npz(softlabel_dir, filename=args.filename)
    if args.limit_cells and args.limit_cells > 0:
        files = files[: int(args.limit_cells)]
    if not files:
        raise FileNotFoundError(f"No {args.filename} found under {softlabel_dir}")

    eval_stride = int(args.eval_stride if args.eval_stride is not None else data_cfg.get("eval_stride", 1))
    include_onehot = bool(state.get("include_profile_onehot", data_cfg.get("include_profile_onehot", True)))
    model_profile_count = len(profile_ids)

    rows_raw: List[Dict[str, Any]] = []
    rows_projected: List[Dict[str, Any]] = []
    rows_primary: List[Dict[str, Any]] = []
    routing_rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    all_true: List[np.ndarray] = []
    all_raw: List[np.ndarray] = []
    all_projected: List[np.ndarray] = []

    pred_primary_dir = eval_dir / "predictions"
    pred_raw_dir = eval_dir / "predictions_raw"
    pred_projected_dir = eval_dir / "predictions_projected"
    pred_both_dir = eval_dir / "predictions_raw_projected"
    for d in (pred_primary_dir, pred_raw_dir, pred_projected_dir, pred_both_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[D16-P5A fixed] softlabel_dir={softlabel_dir}", flush=True)
    print(f"[D16-P5A fixed] model_file={mf}", flush=True)
    print(f"[D16-P5A fixed] files={len(files)}; model_profile_count={model_profile_count}; device={device}; primary_mode={args.primary_mode}", flush=True)

    for default_i, npz_path in enumerate(files):
        pid = profile_id_from_path(npz_path, softlabel_dir)
        try:
            meta = _profile_meta_from_npz(npz_path, softlabel_dir)
            route_index, route_profile_id, route_reason = route_to_seen_profile(meta, seen)
            prof = load_profile_arrays(npz_path, softlabel_dir)
            X, feature_names = build_features(
                prof,
                route_index,
                model_profile_count,
                include_profile_onehot=include_onehot,
            )
            Y, target_names, actual_slices = build_targets(prof)
            stride = max(1, eval_stride)
            X_eval = X[::stride]
            Y_eval = Y[::stride]
            t_eval = prof["t"][::stride]
            if int(X_eval.shape[1]) != int(state["input_dim"]):
                raise ValueError(
                    f"feature_dim mismatch for {pid}: X_eval.shape[1]={X_eval.shape[1]} but model input_dim={state['input_dim']}. "
                    f"include_profile_onehot={include_onehot}; routed profile_count={model_profile_count}; route_index={route_index}."
                )
            if int(Y_eval.shape[1]) != int(state["output_dim"]):
                raise ValueError(
                    f"target_dim mismatch for {pid}: Y_eval.shape[1]={Y_eval.shape[1]} but model output_dim={state['output_dim']}"
                )
            Y_raw = predict_numpy(
                model,
                X_eval,
                np.asarray(state["x_mean"], dtype=np.float32),
                np.asarray(state["x_std"], dtype=np.float32),
                np.asarray(state["y_mean"], dtype=np.float32),
                np.asarray(state["y_std"], dtype=np.float32),
                device,
                batch_size=int(args.batch_size),
            )
            Y_projected = _apply_projection(Y_raw, target_slices, theta_min=args.theta_min, theta_max=args.theta_max)
            Y_primary = Y_projected if args.primary_mode == "projected" else Y_raw

            safe = _safe_name(pid)
            common_npz = dict(
                t_global_s=t_eval.astype(np.float32),
                y_true=Y_eval.astype(np.float32),
                target_names=np.array(state["target_names"]),
                feature_names=np.array(state["feature_names"]),
                profile_id=np.array(str(pid)),
                routed_profile_index=np.array(int(route_index)),
                routed_profile_id=np.array(str(route_profile_id)),
                route_reason=np.array(str(route_reason)),
                primary_mode=np.array(str(args.primary_mode)),
            )
            np.savez_compressed(pred_primary_dir / f"{safe}_prediction.npz", **common_npz, y_pred=Y_primary.astype(np.float32))
            np.savez_compressed(pred_raw_dir / f"{safe}_raw_prediction.npz", **common_npz, y_pred=Y_raw.astype(np.float32))
            np.savez_compressed(pred_projected_dir / f"{safe}_projected_prediction.npz", **common_npz, y_pred=Y_projected.astype(np.float32))
            np.savez_compressed(
                pred_both_dir / f"{safe}_raw_projected_prediction.npz",
                **common_npz,
                y_pred_raw=Y_raw.astype(np.float32),
                y_pred_projected=Y_projected.astype(np.float32),
            )

            extra = {
                "npz_path": str(npz_path),
                "n_eval": int(Y_eval.shape[0]),
                "eval_stride": int(stride),
                "batch": meta.get("batch", ""),
                "protocol": meta.get("protocol", ""),
                "battery_num": meta.get("battery_num"),
                "routed_profile_index": int(route_index),
                "routed_profile_id": str(route_profile_id),
                "route_reason": str(route_reason),
                "seen_exact": bool(route_reason == "exact_profile_id"),
            }
            row_raw = _metrics_row(pid, "raw", Y_eval, Y_raw, target_slices, extra)
            row_projected = _metrics_row(pid, "projected", Y_eval, Y_projected, target_slices, extra)
            row_primary = _metrics_row(pid, args.primary_mode, Y_eval, Y_primary, target_slices, extra)
            rows_raw.append(row_raw)
            rows_projected.append(row_projected)
            rows_primary.append(row_primary)
            routing_rows.append({"profile_id": pid, **extra})
            all_true.append(Y_eval)
            all_raw.append(Y_raw)
            all_projected.append(Y_projected)
            print(
                f"[D16-P5A fixed] {pid}: route={route_index} ({route_reason}); "
                f"raw theta_out={row_raw.get('pred_theta_outside_fraction'):.6g}; "
                f"proj phis_c_mae={row_projected.get('phis_c_mae'):.6g}; proj theta_a_mae={row_projected.get('theta_a_mae'):.6g}",
                flush=True,
            )
        except Exception as exc:
            failures.append(f"{npz_path}: {exc!r}")
            print(f"[D16-P5A fixed] FAILURE {npz_path}: {exc!r}", flush=True)

    if not all_true:
        summary = {
            "stage": "D16-P5A fixed existing-model transfer eval",
            "operational_status": "FAIL",
            "overall_status": "FAIL",
            "reason": "no predictions generated",
            "failures": failures,
        }
        _write_json(summary, run_dir / "D16_P5A_FIXED_SCORECARD.json")
        return 2

    YT = np.concatenate(all_true, axis=0)
    YR = np.concatenate(all_raw, axis=0)
    YP = np.concatenate(all_projected, axis=0)
    raw_global = compute_rg_metrics(YT, YR, target_slices)
    projected_global = compute_rg_metrics(YT, YP, target_slices)
    primary_global = projected_global if args.primary_mode == "projected" else raw_global
    raw_score = thresholds_status(dict(raw_global), thresholds)
    projected_score = thresholds_status(dict(projected_global), thresholds)
    primary_score = thresholds_status(dict(primary_global), thresholds)

    rows_combined = rows_raw + rows_projected
    _write_csv(rows_combined, eval_dir / "D16_P5A_METRICS_BY_PROFILE.csv")
    _write_json(rows_combined, eval_dir / "D16_P5A_METRICS_BY_PROFILE.json")
    _write_csv(rows_primary, eval_dir / "D15_P1_METRICS_BY_PROFILE.csv")
    _write_json(rows_primary, eval_dir / "D15_P1_METRICS_BY_PROFILE.json")
    _write_csv(rows_primary, eval_dir / "D15_P2_METRICS_BY_PROFILE.csv")
    _write_json(rows_primary, eval_dir / "D15_P2_METRICS_BY_PROFILE.json")
    _write_csv(routing_rows, eval_dir / "D16_P5A_ROUTING_TABLE.csv")
    _write_json(routing_rows, eval_dir / "D16_P5A_ROUTING_TABLE.json")
    _write_csv(_aggregate_group(rows_combined, ["projection_mode", "batch"]), eval_dir / "D16_P5A_BATCH_METRICS.csv")
    _write_json(_aggregate_group(rows_combined, ["projection_mode", "batch"]), eval_dir / "D16_P5A_BATCH_METRICS.json")
    _write_csv(_aggregate_group(rows_combined, ["projection_mode", "protocol"]), eval_dir / "D16_P5A_PROTOCOL_METRICS.csv")
    _write_json(_aggregate_group(rows_combined, ["projection_mode", "protocol"]), eval_dir / "D16_P5A_PROTOCOL_METRICS.json")
    _write_csv([{"failure": f} for f in failures], eval_dir / "D16_P5A_FAILURES.csv")

    eval_summary = {
        "stage": "D16-P5A fixed ALL55 existing-model transfer full-profile evaluation",
        "softlabel_dir": str(softlabel_dir),
        "model_file": str(mf),
        "model_dir": str(model_dir),
        "run_dir": str(run_dir),
        "out_dir": str(eval_dir),
        "profile_count_discovered": len(files),
        "profile_count_predicted": len(rows_projected),
        "model_seen_profile_count": model_profile_count,
        "primary_mode_for_d15_audit": args.primary_mode,
        "eval_stride": int(eval_stride),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "raw_global_metrics": raw_global,
        "projected_global_metrics": projected_global,
        "global_metrics": primary_global,
        "raw_scorecard": raw_score,
        "projected_scorecard": projected_score,
        "scorecard": primary_score,
        "overall_status": primary_score.get("overall_status", "REVIEW"),
        "failures": failures,
        "notes": [
            "This is D16-P5A transfer evaluation of an existing D15 model on ALL55 P2Dlite-RG labels.",
            "Unseen ALL55 profiles are routed into the trained model's finite profile-onehot space; see D16_P5A_ROUTING_TABLE.csv.",
            "Projected predictions clip only theta_a/theta_c channels; phie and phis_c are unchanged.",
        ],
    }
    _write_json(eval_summary, eval_dir / "D16_P5A_EVAL_SUMMARY.json")
    _write_json(eval_summary, eval_dir / "D15_P1_EVAL_SUMMARY.json")
    d15_p2_alias = dict(eval_summary)
    d15_p2_alias["stage_alias"] = "D15-P2-compatible eval summary produced by D16-P5A fixed runner"
    d15_p2_alias["prediction_npz_saved"] = True
    _write_json(d15_p2_alias, eval_dir / "D15_P2_EVAL_SUMMARY.json")

    # Compatibility training summary so old collect_scorecard no longer fails only because this is a transfer run.
    training_compat = {
        "stage": "D16-P5A existing-model transfer - no retraining performed",
        "overall_status": "TRANSFER_MODEL_LOADED",
        "overall_status_sampled_val": "TRANSFER_MODEL_LOADED",
        "model_dir": str(model_dir),
        "model_file": str(mf),
        "source_model_seen_profile_count": model_profile_count,
        "note": "This file is a D16 compatibility stub. D16-P5A evaluates an existing model and should not require new D15-P2 training summary in run-dir.",
    }
    _write_json(training_compat, run_dir / "D15_P2_TRAINING_SUMMARY.json")

    audit_summary: Dict[str, Any] = {}
    if not args.no_internal_audit:
        audit_summary = _run_internal_precision_audit(eval_dir, softlabel_dir, audit_dir, audit_cfg, filename=args.filename)

    # D16-specific scorecard: operational PASS means predictions were generated; transfer PASS/REVIEW is metric-based.
    operational_status = "PASS" if len(rows_projected) == len(files) and not failures else "REVIEW"
    audit_status = audit_summary.get("overall_status", "SKIPPED") if audit_summary else "SKIPPED"
    metric_status = primary_score.get("overall_status", "REVIEW")
    # Treat metric threshold misses as REVIEW for D16 transfer diagnosis; FAIL only means operational failure.
    final_status = "PASS" if operational_status == "PASS" and metric_status == "PASS" and audit_status in {"PASS", "SKIPPED"} else "REVIEW"
    if operational_status != "PASS" or len(rows_projected) == 0:
        final_status = "FAIL"

    scorecard = {
        "stage": "D16-P5A fixed existing-model transfer scorecard",
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
        "audit_dir": str(audit_dir),
        "softlabel_dir": str(softlabel_dir),
        "model_dir": str(model_dir),
        "model_file": str(mf),
        "profile_count_discovered": len(files),
        "profile_count_predicted": len(rows_projected),
        "model_seen_profile_count": model_profile_count,
        "primary_mode": args.primary_mode,
        "operational_status": operational_status,
        "metric_status_primary": metric_status,
        "raw_scorecard_status": raw_score.get("overall_status"),
        "projected_scorecard_status": projected_score.get("overall_status"),
        "precision_audit_status": audit_status,
        "final_status": final_status,
        "raw_global_metrics": raw_global,
        "projected_global_metrics": projected_global,
        "primary_global_metrics": primary_global,
        "primary_scorecard_checks": primary_score.get("checks", []),
        "precision_audit_summary": audit_summary.get("aggregate", {}) if audit_summary else {},
        "precision_audit_checks": audit_summary.get("precision_status", {}).get("checks", []) if audit_summary else [],
        "failures": failures,
        "important_outputs": {
            "primary_predictions_for_d15_audit": str(pred_primary_dir),
            "raw_predictions": str(pred_raw_dir),
            "projected_predictions": str(pred_projected_dir),
            "raw_projected_predictions": str(pred_both_dir),
            "metrics_by_profile": str(eval_dir / "D16_P5A_METRICS_BY_PROFILE.csv"),
            "routing_table": str(eval_dir / "D16_P5A_ROUTING_TABLE.csv"),
            "batch_metrics": str(eval_dir / "D16_P5A_BATCH_METRICS.csv"),
            "protocol_metrics": str(eval_dir / "D16_P5A_PROTOCOL_METRICS.csv"),
            "precision_audit_summary": str(audit_dir / "D15_P2_PRECISION_AUDIT_SUMMARY.json"),
        },
        "interpretation": "PASS means the existing model transferred to ALL55 under the chosen primary prediction mode. REVIEW means predictions were generated but at least one metric/audit threshold requires diagnosis before D16-P5B. FAIL means operational failure such as no predictions or unreadable data.",
    }
    _write_json(scorecard, run_dir / "D16_P5A_FIXED_SCORECARD.json")
    # Also write the requested D16 transfer scorecard filename.
    _write_json(scorecard, run_dir / "D16_P5A_D15P2_TRANSFER_SCORECARD_FIXED.json")

    print("[D16-P5A fixed] final_status:", final_status, flush=True)
    print("[D16-P5A fixed] scorecard:", run_dir / "D16_P5A_FIXED_SCORECARD.json", flush=True)
    print("[D16-P5A fixed] primary predictions:", pred_primary_dir, flush=True)
    print("[D16-P5A fixed] precision audit:", audit_dir, flush=True)
    return 0 if final_status in {"PASS", "REVIEW"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
