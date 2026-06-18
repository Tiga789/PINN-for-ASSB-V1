from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .g1_data import (
    OBS_TIME_KEYS,
    json_dump,
    load_split_records,
    load_semantics_map,
    _semantics_for,
    load_profile_pack,
    _load_npz_dict,
    _find_1d_exact,
    _find_1d_any,
)
from .g3_frozen_audit import (
    read_json,
    torch_load_safe,
    write_csv,
    safe_float,
    build_model_from_checkpoint,
    parse_vocab_from_feature_names,
    resolve_checkpoint_path,
    augment_profile_features,
)
from .g13_trainer import _device_from_arg

CYCLE_KEYS = ["cycle_id", "cycle", "cycle_index", "cycle_number", "cycle_idx"]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class RunningStats:
    n: int = 0
    sum_y: float = 0.0
    sum_y2: float = 0.0
    sum_abs: float = 0.0
    sum_sqerr: float = 0.0
    sum_err: float = 0.0
    y_min: float = field(default_factory=lambda: float("inf"))
    y_max: float = field(default_factory=lambda: float("-inf"))
    pred_min: float = field(default_factory=lambda: float("inf"))
    pred_max: float = field(default_factory=lambda: float("-inf"))

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
        yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        mask = np.isfinite(yt) & np.isfinite(yp)
        if not np.any(mask):
            return
        yt = yt[mask]
        yp = yp[mask]
        err = yp - yt
        self.n += int(yt.size)
        self.sum_y += float(np.sum(yt))
        self.sum_y2 += float(np.sum(yt * yt))
        self.sum_abs += float(np.sum(np.abs(err)))
        self.sum_sqerr += float(np.sum(err * err))
        self.sum_err += float(np.sum(err))
        self.y_min = min(self.y_min, float(np.min(yt)))
        self.y_max = max(self.y_max, float(np.max(yt)))
        self.pred_min = min(self.pred_min, float(np.min(yp)))
        self.pred_max = max(self.pred_max, float(np.max(yp)))

    def merge(self, other: "RunningStats") -> None:
        self.n += other.n
        self.sum_y += other.sum_y
        self.sum_y2 += other.sum_y2
        self.sum_abs += other.sum_abs
        self.sum_sqerr += other.sum_sqerr
        self.sum_err += other.sum_err
        self.y_min = min(self.y_min, other.y_min)
        self.y_max = max(self.y_max, other.y_max)
        self.pred_min = min(self.pred_min, other.pred_min)
        self.pred_max = max(self.pred_max, other.pred_max)

    def metrics(self) -> Dict[str, Any]:
        if self.n <= 0:
            return {"n_points": 0, "mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "nmae": float("nan"), "nrmse": float("nan"), "bias": float("nan"), "target_range": float("nan"), "target_std": float("nan")}
        mae = self.sum_abs / self.n
        rmse = math.sqrt(max(self.sum_sqerr / self.n, 0.0))
        mean_y = self.sum_y / self.n
        sst = self.sum_y2 - self.n * mean_y * mean_y
        r2 = 1.0 - self.sum_sqerr / sst if sst > 1e-18 else float("nan")
        target_range = self.y_max - self.y_min
        target_std = math.sqrt(max(sst / self.n, 0.0)) if sst >= 0 else float("nan")
        denom = target_range if math.isfinite(target_range) and target_range > 1e-12 else float("nan")
        return {
            "n_points": int(self.n),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "nmae": float(mae / denom) if math.isfinite(denom) else float("nan"),
            "nrmse": float(rmse / denom) if math.isfinite(denom) else float("nan"),
            "bias": float(self.sum_err / self.n),
            "target_range": float(target_range),
            "target_std": float(target_std),
            "target_min": float(self.y_min),
            "target_max": float(self.y_max),
            "pred_min": float(self.pred_min),
            "pred_max": float(self.pred_max),
        }


def _as_str(x: Any) -> str:
    try:
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="replace")
        return str(x)
    except Exception:
        return ""


def _to_1d_maybe_numeric(x: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x).reshape(-1)
        if arr.size == 0:
            return None
        if arr.dtype.kind in {"U", "S", "O"}:
            vals: List[float] = []
            for v in arr:
                try:
                    vals.append(float(_as_str(v).strip()))
                except Exception:
                    return arr
            return np.asarray(vals)
        return arr
    except Exception:
        return None


def _interp_nearest(src_values: np.ndarray, src_t: Optional[np.ndarray], target_t: np.ndarray) -> np.ndarray:
    src_values = np.asarray(src_values).reshape(-1)
    target_t = np.asarray(target_t, dtype=np.float64).reshape(-1)
    if src_values.size == target_t.size:
        return src_values
    if src_values.size == 0:
        return np.zeros(target_t.size, dtype=np.int32)
    if src_t is None or np.asarray(src_t).reshape(-1).size != src_values.size:
        src_x = np.linspace(0.0, 1.0, src_values.size)
        tgt_x = np.linspace(0.0, 1.0, target_t.size)
        pos = np.searchsorted(src_x, tgt_x, side="left")
        pos = np.clip(pos, 0, src_values.size - 1)
        return src_values[pos]
    src_t = np.asarray(src_t, dtype=np.float64).reshape(-1)
    order = np.argsort(src_t)
    src_t = src_t[order]
    src_values = src_values[order]
    pos = np.searchsorted(src_t, target_t, side="left")
    pos = np.clip(pos, 0, src_values.size - 1)
    prev = np.clip(pos - 1, 0, src_values.size - 1)
    choose_prev = np.abs(target_t - src_t[prev]) <= np.abs(target_t - src_t[pos])
    pos = np.where(choose_prev, prev, pos)
    return src_values[pos]


def load_cycle_ids_for_profile(profile: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = int(profile.t_global_s.size)
    target_t = np.asarray(profile.t_global_s, dtype=np.float32).reshape(-1)
    details: Dict[str, Any] = {"n_time": n, "cycle_source": "none"}
    for source_name, path in [("softlabel", profile.softlabel_npz), ("replay", profile.replay_npz)]:
        if not path or not Path(path).exists():
            continue
        d = _load_npz_dict(path, list(set(CYCLE_KEYS + OBS_TIME_KEYS)))
        cyc_key = ""
        cyc_arr = None
        for k in CYCLE_KEYS:
            if k in d:
                arr = _to_1d_maybe_numeric(d[k])
                if arr is not None and arr.size > 0:
                    cyc_key = k
                    cyc_arr = arr
                    break
        if cyc_arr is None:
            continue
        if cyc_arr.size == n:
            details.update({"cycle_source": source_name, "cycle_key": cyc_key, "cycle_mode": "exact"})
            return cyc_arr, details
        tk, tt = _find_1d_exact(d, OBS_TIME_KEYS, cyc_arr.size)
        if tt is None:
            _, tt_any = _find_1d_any(d, OBS_TIME_KEYS)
            tt = tt_any if tt_any is not None and tt_any.size == cyc_arr.size else None
        mapped = _interp_nearest(cyc_arr, tt, target_t)
        details.update({"cycle_source": source_name, "cycle_key": cyc_key, "cycle_mode": "nearest", "source_n": int(cyc_arr.size)})
        return mapped, details
    details.update({"cycle_source": "filled_all", "cycle_key": "", "cycle_mode": "single_all"})
    return np.zeros(n, dtype=np.int32), details


def cycle_ranges(cycles: np.ndarray, max_items: int = 32) -> str:
    vals: List[int] = []
    for v in np.asarray(cycles).reshape(-1):
        try:
            vals.append(int(float(_as_str(v))))
        except Exception:
            pass
    if not vals:
        return ""
    vals = sorted(set(vals))
    ranges: List[str] = []
    s = p = vals[0]
    for x in vals[1:]:
        if x == p + 1:
            p = x
        else:
            ranges.append(str(s) if s == p else f"{s}-{p}")
            s = p = x
    ranges.append(str(s) if s == p else f"{s}-{p}")
    return ",".join(ranges[:max_items]) + (f",...(+{len(ranges)-max_items} ranges)" if len(ranges) > max_items else "")


def choose_records(records: Sequence[Mapping[str, Any]], splits: Sequence[str], include_flagged: bool, limit: int = 0) -> List[Dict[str, Any]]:
    split_set = {str(s) for s in splits}
    use_all = "all" in split_set
    out: List[Dict[str, Any]] = []
    for r in records:
        sp = str(r.get("split") or "")
        is_flagged = bool(r.get("is_flagged_probe")) or sp == "flagged_probe"
        if is_flagged and not include_flagged:
            continue
        if use_all or sp in split_set:
            out.append(dict(r))
    if limit and int(limit) > 0:
        out = out[: int(limit)]
    return out


def predict_denorm(model: torch.nn.Module, X: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, device: torch.device) -> np.ndarray:
    xs = x_std.copy()
    xs[~np.isfinite(xs) | (np.abs(xs) < 1e-8)] = 1.0
    Xn = ((X - x_mean.reshape(1, -1)) / xs.reshape(1, -1)).astype(np.float32)
    Xn[~np.isfinite(Xn)] = 0.0
    with torch.no_grad():
        yp = model(torch.as_tensor(Xn, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return (yp * y_std.reshape(1, -1) + y_mean.reshape(1, -1)).astype(np.float32)


def _merge_stats(dst: Dict[Tuple[str, str, str], RunningStats], key: Tuple[str, str, str], yt: np.ndarray, yp: np.ndarray) -> None:
    if key not in dst:
        dst[key] = RunningStats()
    dst[key].update(yt, yp)


def evaluate_profile_full(
    model: torch.nn.Module,
    profile: Any,
    X_aug: np.ndarray,
    cycles: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    device: torch.device,
    target_slices: Mapping[str, Tuple[int, int]],
    chunk_points: int,
    cycle_metrics: bool = True,
    save_pred_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[Tuple[str, str, str], RunningStats], Optional[Path]]:
    profile_stats: Dict[str, RunningStats] = {k: RunningStats() for k in target_slices}
    cycle_stats: Dict[Tuple[str, str], RunningStats] = {}
    group_stats: Dict[Tuple[str, str, str], RunningStats] = {}
    n = int(profile.targets.shape[0])
    chunk_points = max(1, int(chunk_points))
    pred_chunks: Dict[str, List[np.ndarray]] = {k: [] for k in target_slices} if save_pred_path is not None else {}
    model.eval()
    for i0 in range(0, n, chunk_points):
        i1 = min(n, i0 + chunk_points)
        pred = predict_denorm(model, X_aug[i0:i1], x_mean, x_std, y_mean, y_std, device)
        ytrue = profile.targets[i0:i1]
        cyc = np.asarray(cycles[i0:i1]).reshape(-1)
        for target, (a, b) in target_slices.items():
            yt = ytrue[:, a:b]
            yp = pred[:, a:b]
            profile_stats[target].update(yt, yp)
            for group_type, group_value in [
                ("all", "all"),
                ("split", profile.split),
                ("protocol", profile.protocol),
                ("semantic_branch", profile.branch),
                ("protocol_branch", f"{profile.protocol}::{profile.branch}"),
            ]:
                _merge_stats(group_stats, (group_type, str(group_value), target), yt, yp)
            if cycle_metrics:
                for cv in np.unique(cyc):
                    mask = cyc == cv
                    ckey = (_as_str(cv), target)
                    if ckey not in cycle_stats:
                        cycle_stats[ckey] = RunningStats()
                    cycle_stats[ckey].update(yt[mask], yp[mask])
            if save_pred_path is not None:
                pred_chunks[target].append(yp.astype(np.float32))
    profile_rows: List[Dict[str, Any]] = []
    for target, st in profile_stats.items():
        profile_rows.append({
            "split": profile.split,
            "canonical_cell_uid": profile.canonical_cell_uid,
            "cell_uid": profile.cell_uid,
            "protocol": profile.protocol,
            "semantic_branch": profile.branch,
            "target": target,
            **st.metrics(),
        })
    cycle_rows: List[Dict[str, Any]] = []
    if cycle_metrics:
        for (cycle_id, target), st in sorted(cycle_stats.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
            cycle_rows.append({
                "split": profile.split,
                "canonical_cell_uid": profile.canonical_cell_uid,
                "cell_uid": profile.cell_uid,
                "protocol": profile.protocol,
                "semantic_branch": profile.branch,
                "cycle_id": cycle_id,
                "target": target,
                **st.metrics(),
            })
    written_path: Optional[Path] = None
    if save_pred_path is not None:
        arrays: Dict[str, Any] = {
            "t_global_s": profile.t_global_s.astype(np.float32),
            "cycle_id": np.asarray(cycles),
            "canonical_cell_uid": np.array(profile.canonical_cell_uid),
            "cell_uid": np.array(profile.cell_uid),
            "protocol": np.array(profile.protocol),
            "semantic_branch": np.array(profile.branch),
        }
        for target, chunks in pred_chunks.items():
            arrays[f"{target}_pred"] = np.concatenate(chunks, axis=0).astype(np.float32) if chunks else np.zeros((0, 1), dtype=np.float32)
            a, b = target_slices[target]
            arrays[f"{target}_true_report_only"] = profile.targets[:, a:b].astype(np.float32)
        save_pred_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_pred_path, **arrays)
        written_path = save_pred_path
    return profile_rows, cycle_rows, group_stats, written_path


def stats_rows_from_group(group_stats: Mapping[Tuple[str, str, str], RunningStats]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (group_type, group_value, target), st in sorted(group_stats.items()):
        rows.append({"group_type": group_type, "group_value": group_value, "target": target, **st.metrics()})
    return rows


def aggregate_profile_target_rows(rows: Sequence[Mapping[str, Any]], group_keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[float]] = {}
    for r in rows:
        key = tuple(r.get(k, "") for k in group_keys)
        v = safe_float(r.get("r2"))
        if math.isfinite(v):
            groups.setdefault(key, []).append(v)
    out: List[Dict[str, Any]] = []
    for key, vals in sorted(groups.items()):
        row = {k: key[i] for i, k in enumerate(group_keys)}
        row.update({
            "profile_target_count": len(vals),
            "mean_r2": float(np.mean(vals)) if vals else float("nan"),
            "min_r2": float(np.min(vals)) if vals else float("nan"),
            "max_r2": float(np.max(vals)) if vals else float("nan"),
        })
        out.append(row)
    return out


def find_worst(rows: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    finite = [dict(r) for r in rows if math.isfinite(safe_float(r.get("r2")))]
    if not finite:
        return None
    return min(finite, key=lambda r: safe_float(r.get("r2"), 1e99))


def run_g6_full_cycle_audit(
    split_manifest: str | Path,
    g0_profile_semantics_csv: str | Path,
    candidate_g21_dir: str | Path,
    candidate_g21_summary: str | Path,
    out_dir: str | Path,
    config: Mapping[str, Any],
    checkpoint_path: str | Path = "",
    splits: Sequence[str] = ("all",),
    include_flagged_probe: bool = True,
    profile_limit: int = 0,
    max_time_points: int = 0,
    time_window_s: float = 0.0,
    predict_batch_size: int = 8192,
    save_predictions: str = "none",
    cycle_metrics: bool = True,
    device_arg: str = "auto",
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidate = read_json(candidate_g21_summary, default={}) or {}
    candidate_ready = bool(candidate.get("g3_ready")) and str(candidate.get("status")) == "PASS"
    if bool(config.get("require_g21_ready", True)) and not candidate_ready:
        summary = {"protocol": "D17-G6_FULL_ALLCELL_ALLCYCLE_AUDIT", "created_at_utc": utc_now(), "status": "BLOCKED", "promotion_status": "BLOCKED", "blockers": ["candidate G2.1 summary is not PASS/g3_ready=true"], "candidate_g21_summary": str(candidate_g21_summary), "candidate_status": candidate.get("status"), "candidate_g3_ready": candidate.get("g3_ready")}
        json_dump(summary, out / "D17_G6_FULL_ALLCELL_ALLCYCLE_AUDIT_SUMMARY.json")
        return summary
    ckpt_path = resolve_checkpoint_path(str(checkpoint_path or ""), candidate, candidate_g21_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Cannot find G2.1 checkpoint: {ckpt_path}")
    ckpt = torch_load_safe(ckpt_path, map_location="cpu")
    device = _device_from_arg(device_arg)
    model = build_model_from_checkpoint(ckpt, device)
    feature_names = list(ckpt.get("feature_names") or [])
    local_input_dim = int(ckpt.get("local_input_dim", 0))
    _, protocol_vocab, branch_vocab = parse_vocab_from_feature_names(feature_names, local_input_dim)
    if not protocol_vocab or not branch_vocab:
        raise ValueError("Could not parse protocol/branch vocab from checkpoint feature_names")
    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std = np.asarray(ckpt["y_std"], dtype=np.float32)
    target_slices = {str(k): (int(v[0]), int(v[1])) for k, v in dict(ckpt.get("target_slices") or {}).items()}
    records, manifest = load_split_records(split_manifest)
    sem_map = load_semantics_map(g0_profile_semantics_csv)
    selected_records = choose_records(records, splits=splits, include_flagged=include_flagged_probe, limit=int(profile_limit))
    if not selected_records:
        raise ValueError("No records selected for G6 full-cycle audit")

    profile_rows: List[Dict[str, Any]] = []
    cycle_rows: List[Dict[str, Any]] = []
    load_failures: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    pred_manifest: List[Dict[str, Any]] = []
    global_stats: Dict[Tuple[str, str, str], RunningStats] = {}
    total_points = 0
    start = time.time()
    for idx, rec in enumerate(selected_records):
        canonical = str(rec.get("canonical_cell_uid") or rec.get("cell_uid") or f"profile_{idx}")
        try:
            prof = load_profile_pack(rec, _semantics_for(rec, sem_map), protocol_vocab, branch_vocab, int(max_time_points), float(time_window_s))
            cycles, cycle_info = load_cycle_ids_for_profile(prof)
            X_aug, finfo, aug_names = augment_profile_features(prof)
            if X_aug.shape[1] != x_mean.size:
                raise ValueError(f"feature dim mismatch for {canonical}: X_aug={X_aug.shape[1]}, checkpoint={x_mean.size}")
            if list(aug_names) != feature_names:
                raise ValueError(f"feature name mismatch for {canonical}; refusing silent schema drift")
            total_points += int(prof.targets.shape[0])
            feature_rows.append({"profile_index": idx, "canonical_cell_uid": prof.canonical_cell_uid, "cell_uid": prof.cell_uid, "split": prof.split, "protocol": prof.protocol, "semantic_branch": prof.branch, "n_time": int(prof.targets.shape[0]), "cycle_ranges": cycle_ranges(cycles), **cycle_info, **{f"obs_{k}": v for k, v in finfo.items() if isinstance(v, (int, float, str, bool))}})
            pred_path: Optional[Path] = None
            if str(save_predictions).lower() == "compressed_npz":
                safe_name = prof.canonical_cell_uid.replace("\\", "_").replace("/", "_")
                pred_path = out / "predictions" / str(prof.split) / f"D17_G6_FULL_{idx:03d}_{safe_name}_PRED.npz"
            pr, cr, gs, written = evaluate_profile_full(model, prof, X_aug, cycles, x_mean, x_std, y_mean, y_std, device, target_slices, int(predict_batch_size), cycle_metrics=cycle_metrics, save_pred_path=pred_path)
            profile_rows.extend(pr)
            cycle_rows.extend(cr)
            for k, st in gs.items():
                global_stats.setdefault(k, RunningStats()).merge(st)
            if written is not None:
                pred_manifest.append({"profile_index": idx, "split": prof.split, "canonical_cell_uid": prof.canonical_cell_uid, "pred_npz": str(written), "n_time": int(prof.targets.shape[0]), "cycle_ranges": cycle_ranges(cycles)})
            del prof, X_aug, cycles
        except Exception as e:
            load_failures.append({"profile_index": idx, "canonical_cell_uid": canonical, "cell_uid": rec.get("cell_uid"), "split": rec.get("split"), "protocol": rec.get("protocol"), "softlabel_npz": rec.get("softlabel_npz"), "replay_npz": rec.get("replay_npz"), "error": repr(e)})

    elapsed = max(time.time() - start, 1e-9)
    group_rows = stats_rows_from_group(global_stats)
    prof_aggs = []
    for keys in (["split"], ["split", "target"], ["protocol"], ["semantic_branch"], ["protocol", "semantic_branch"]):
        prof_aggs.extend(aggregate_profile_target_rows(profile_rows, keys))
    write_csv(profile_rows, out / "D17_G6_PROFILE_TARGET_METRICS.csv")
    write_csv(cycle_rows, out / "D17_G6_CYCLE_TARGET_METRICS.csv")
    write_csv(group_rows, out / "D17_G6_RAW_WEIGHTED_GROUP_TARGET_METRICS.csv")
    write_csv(prof_aggs, out / "D17_G6_PROFILE_R2_AGGREGATES.csv")
    write_csv(feature_rows, out / "D17_G6_FEATURE_AND_CYCLE_AUDIT.csv")
    write_csv(load_failures, out / "D17_G6_LOAD_FAILURES.csv")
    write_csv(pred_manifest, out / "D17_G6_PREDICTION_MANIFEST.csv")

    all_r2 = [safe_float(r.get("r2")) for r in profile_rows]
    all_r2 = [v for v in all_r2 if math.isfinite(v)]
    all_mean = float(np.mean(all_r2)) if all_r2 else float("nan")
    all_min = float(np.min(all_r2)) if all_r2 else float("nan")
    target_summary: Dict[str, Any] = {}
    for target in sorted({str(r.get("target")) for r in profile_rows}):
        vals = [safe_float(r.get("r2")) for r in profile_rows if str(r.get("target")) == target]
        vals = [v for v in vals if math.isfinite(v)]
        if vals:
            target_summary[f"{target}_profile_r2_mean"] = float(np.mean(vals))
            target_summary[f"{target}_profile_r2_min"] = float(np.min(vals))
    mean_gate = float(config.get("all_profile_target_mean_r2_threshold", 0.95))
    min_gate = float(config.get("all_profile_target_min_r2_threshold", 0.90))
    blockers: List[str] = []
    if load_failures:
        blockers.append(f"{len(load_failures)} profiles failed to load/evaluate")
    if all_mean < mean_gate or all_min < min_gate:
        blockers.append(f"all-cell all-cycle profile-target R2 below gate: mean={all_mean:.6g}, min={all_min:.6g}")
    status = "PASS" if not load_failures else "REVIEW"
    promotion_status = "PASS" if status == "PASS" and not blockers else "REVIEW"
    summary: Dict[str, Any] = {
        "protocol": "D17-G6_FULL_ALLCELL_ALLCYCLE_AUDIT",
        "created_at_utc": utc_now(),
        "status": status,
        "promotion_status": promotion_status,
        "full_cycle_all55_ready": bool(promotion_status == "PASS"),
        "recommendation": "USE_G6_AS_FULL_ALLCELL_ALLCYCLE_EVIDENCE" if promotion_status == "PASS" else "REVIEW_G6_FAILURES_BEFORE_FULL_CYCLE_CLAIM",
        "blockers": blockers,
        "purpose": "Frozen full soft-label time-grid audit of the D17-G generator surrogate, replacing the earlier sampled-window-only evidence for all-cell/all-cycle claims.",
        "policy": {"training_performed": False, "checkpoint_selection_performed": False, "candidate_modified": False, "train_cell_softlabels_were_used_upstream": True, "validation_and_frozen_test_softlabels_are_report_only_in_G6": True, "frozen_test_feedback_used_for_model_update": False, "save_predictions_mode": str(save_predictions)},
        "candidate": {"candidate_summary": str(candidate_g21_summary), "candidate_protocol": candidate.get("protocol"), "candidate_status": candidate.get("status"), "candidate_g3_ready": candidate.get("g3_ready"), "checkpoint": str(ckpt_path), "checkpoint_best_epoch": ckpt.get("best_epoch")},
        "dataset": {"manifest_hash_sha256": manifest.get("manifest_hash_sha256"), "record_counts": manifest.get("counts"), "selected_record_count": len(selected_records), "evaluated_profile_count": len({str(r.get("canonical_cell_uid")) for r in profile_rows}), "load_failure_count": len(load_failures), "total_time_points_evaluated": int(total_points), "target_dim": int(y_mean.size), "augmented_feature_dim": int(x_mean.size), "splits_requested": list(splits), "include_flagged_probe": bool(include_flagged_probe), "max_time_points": int(max_time_points), "time_window_s": float(time_window_s)},
        "runtime": {"elapsed_s": float(elapsed), "points_per_second": float(total_points / elapsed), "device": str(device), "predict_batch_size": int(predict_batch_size)},
        "all_profile_target_r2_mean": all_mean,
        "all_profile_target_r2_min": all_min,
        "per_target_profile_r2_summary": target_summary,
        "worst_profile_target": find_worst(profile_rows),
        "worst_cycle_target": find_worst(cycle_rows),
        "raw_weighted_global_target_metrics": [r for r in group_rows if r.get("group_type") == "all"],
        "files": {"summary_json": str(out / "D17_G6_FULL_ALLCELL_ALLCYCLE_AUDIT_SUMMARY.json"), "scorecard_json": str(out / "D17_G6_SCORECARD.json"), "profile_target_metrics_csv": str(out / "D17_G6_PROFILE_TARGET_METRICS.csv"), "cycle_target_metrics_csv": str(out / "D17_G6_CYCLE_TARGET_METRICS.csv"), "raw_weighted_group_target_metrics_csv": str(out / "D17_G6_RAW_WEIGHTED_GROUP_TARGET_METRICS.csv"), "profile_r2_aggregates_csv": str(out / "D17_G6_PROFILE_R2_AGGREGATES.csv"), "feature_and_cycle_audit_csv": str(out / "D17_G6_FEATURE_AND_CYCLE_AUDIT.csv"), "load_failures_csv": str(out / "D17_G6_LOAD_FAILURES.csv"), "prediction_manifest_csv": str(out / "D17_G6_PREDICTION_MANIFEST.csv")},
    }
    json_dump(summary, out / "D17_G6_FULL_ALLCELL_ALLCYCLE_AUDIT_SUMMARY.json")
    scorecard = {"protocol": summary["protocol"], "status": summary["status"], "promotion_status": summary["promotion_status"], "full_cycle_all55_ready": summary["full_cycle_all55_ready"], "blockers": blockers, "dataset": summary["dataset"], "all_profile_target_r2_mean": all_mean, "all_profile_target_r2_min": all_min, "per_target_profile_r2_summary": target_summary, "worst_profile_target": summary["worst_profile_target"], "worst_cycle_target": summary["worst_cycle_target"], "policy": summary["policy"]}
    json_dump(scorecard, out / "D17_G6_SCORECARD.json")
    return summary
