from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from gv1.d17_g.g1_data import (
    G1Dataset,
    ProfilePack,
    build_g1_dataset,
    json_dump,
    json_load,
)
from gv1.d17_g.g1_metrics import aggregate_profile_rows, group_metrics, profile_metrics, r2_score
from gv1.d17_g.g1_model import GeneratorSurrogateMLP

GENERATOR_FILES = [
    "scripts/d15_p0_generate_p2dlite_rg_softlabels.py",
    "scripts/d15_p3c_generate_batch2_15cell_rg_softlabels.py",
    "scripts/d15_p3c_generate_batch2_rg_softlabels.py",
    "scripts/d15_p4b_generate_ready18_rg_softlabels.py",
    "scripts/d15_p4d_full_generate_one_rg_softlabel.py",
    "scripts/d15_p4d_generate_one_smoke_profile.py",
    "gv1/p2dlite_rg/radial_solver.py",
    "gv1/p2dlite_rg/io_utils.py",
    "gv1/p2dlite_rg/data.py",
    "gv1/p2dlite_rg/model.py",
    "gv1/p2dlite_rg/train_eval.py",
]

PATTERN_GROUPS = {
    "d15_p0_rg_repair_from_source": [
        "generate_rg_profile",
        "source_p2dlite_v1_key_a",
        "source_p2dlite_v1_key_c",
        "phis_c_voltage_preserved_from_source",
        "Preserve voltage and phi labels",
        "source_flux_method_a",
        "source_flux_method_c",
    ],
    "d15_p4d_current_integral_branch": [
        "capacity_scale_Ah",
        "theta_positive_initial",
        "theta_negative_initial",
        "phis_c_soft",
        "phie_ohmic_scale_V_per_A",
        "cbar_c",
        "cbar_a",
    ],
    "radial_fvm_generator_core": [
        "ElectrodeRGParams",
        "infer_surface_flux_from_cbar",
        "generate_rg_profile",
        "zero",
        "volume_weights",
        "surface_center",
    ],
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    if not keys:
        keys = ["empty"]
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))


def scan_generator_code(project_root: str | Path) -> Dict[str, Any]:
    root = Path(project_root)
    files = []
    group_hits = {g: 0 for g in PATTERN_GROUPS}
    missing_files: List[str] = []
    for rel in GENERATOR_FILES:
        p = root / rel
        item: Dict[str, Any] = {"relative_path": rel, "exists": p.exists()}
        if not p.exists():
            missing_files.append(rel)
            item.update({"size_bytes": 0, "pattern_hits": {}})
            files.append(item)
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            text = ""
            item["read_error"] = repr(exc)
        item["size_bytes"] = int(p.stat().st_size)
        item["line_count"] = int(text.count("\n") + 1) if text else 0
        ph: Dict[str, Dict[str, Any]] = {}
        for group, needles in PATTERN_GROUPS.items():
            count = sum(text.count(n) for n in needles)
            present = count > 0
            ph[group] = {"present": present, "count": int(count)}
            if present:
                group_hits[group] += 1
        item["pattern_hits"] = ph
        files.append(item)
    missing_groups = [g for g, c in group_hits.items() if c == 0]
    status = "PASS" if not missing_files and not missing_groups else "REVIEW"
    return {
        "status": status,
        "missing_files": missing_files,
        "missing_pattern_groups": missing_groups,
        "pattern_group_file_hit_counts": group_hits,
        "files": files,
    }


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _predict_np(model: torch.nn.Module, Xn: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, device: torch.device, batch_size: int = 16384) -> np.ndarray:
    model.eval()
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, Xn.shape[0], batch_size):
            xb = torch.as_tensor(Xn[i:i+batch_size], dtype=torch.float32, device=device)
            predn = model(xb).detach().cpu().numpy().astype(np.float32)
            outs.append(predn * y_std[None, :] + y_mean[None, :])
    return np.concatenate(outs, axis=0).astype(np.float32) if outs else np.zeros((0, y_mean.size), dtype=np.float32)


def _split_profile_predictions(pred_all: np.ndarray, profiles: Sequence[ProfilePack]) -> List[np.ndarray]:
    out = []
    cursor = 0
    for p in profiles:
        n = int(p.targets.shape[0])
        out.append(pred_all[cursor:cursor+n])
        cursor += n
    return out


def _feature_target_stats(dataset: G1Dataset, profiles: Sequence[ProfilePack]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    target_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    Y = np.concatenate([p.targets for p in profiles], axis=0) if profiles else dataset.Y_train
    X = np.concatenate([p.features for p in profiles], axis=0) if profiles else dataset.X_train
    for key, (a, b) in dataset.target_slices.items():
        arr = Y[:, a:b].astype(np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        row: Dict[str, Any] = {"target": key, "dim": int(b-a), "finite_count": int(arr.size)}
        if arr.size:
            row.update({
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p01": float(np.percentile(arr, 1)),
                "p99": float(np.percentile(arr, 99)),
                "range": float(np.max(arr) - np.min(arr)),
            })
        target_rows.append(row)
    for i, name in enumerate(dataset.feature_names):
        arr = X[:, i].astype(np.float64)
        arr = arr[np.isfinite(arr)]
        row = {"feature": name, "finite_count": int(arr.size)}
        if arr.size:
            row.update({"mean": float(np.mean(arr)), "std": float(np.std(arr)), "min": float(np.min(arr)), "max": float(np.max(arr))})
        feature_rows.append(row)
    return target_rows, feature_rows


def _estimate_csmax_consistency(prof: ProfilePack) -> Dict[str, Any]:
    out: Dict[str, Any] = {"canonical_cell_uid": prof.canonical_cell_uid, "semantic_branch": prof.branch}
    for electrode in ["a", "c"]:
        cs_key = f"cs_{electrode}"
        th_key = f"theta_{electrode}"
        if cs_key not in prof.target_slices or th_key not in prof.target_slices:
            continue
        ca, cb = prof.target_slices[cs_key]
        ta, tb = prof.target_slices[th_key]
        cs = prof.targets[:, ca:cb].astype(np.float64).reshape(-1)
        th = prof.targets[:, ta:tb].astype(np.float64).reshape(-1)
        mask = np.isfinite(cs) & np.isfinite(th) & (np.abs(th) > 1e-8)
        if mask.sum() < 10:
            out[f"csmax_est_{electrode}"] = float("nan")
            out[f"cs_theta_relerr_median_{electrode}"] = float("nan")
            continue
        ratio = cs[mask] / th[mask]
        ratio = ratio[np.isfinite(ratio)]
        csmax = float(np.median(ratio)) if ratio.size else float("nan")
        if math.isfinite(csmax) and abs(csmax) > 1e-12:
            relerr = np.abs(cs[mask] - th[mask] * csmax) / max(abs(csmax), 1e-12)
            out[f"csmax_est_{electrode}"] = csmax
            out[f"cs_theta_relerr_median_{electrode}"] = float(np.median(relerr))
            out[f"cs_theta_relerr_p99_{electrode}"] = float(np.percentile(relerr, 99))
        else:
            out[f"csmax_est_{electrode}"] = csmax
            out[f"cs_theta_relerr_median_{electrode}"] = float("nan")
    return out


def _profile_time_alignment_rows(profiles: Sequence[ProfilePack]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in profiles:
        si = p.source_info or {}
        rows.append({
            "split": p.split,
            "canonical_cell_uid": p.canonical_cell_uid,
            "cell_uid": p.cell_uid,
            "protocol": p.protocol,
            "semantic_branch": p.branch,
            "n_points_sampled": int(p.targets.shape[0]),
            "target_grid_policy": si.get("target_grid_policy", ""),
            "n_target_softlabel": si.get("n_target_softlabel", ""),
            "n_replay_time_any": si.get("n_replay_time_any", ""),
            "t_key": si.get("t_key", ""),
            "I_key": si.get("I_key", ""),
            "V_key": si.get("V_key", ""),
            "T_key": si.get("T_key", ""),
            "phie_source_semantics": si.get("phie_source_semantics", ""),
            "phis_c_source_semantics": si.get("phis_c_source_semantics", ""),
        })
    return rows


def make_group_weights(target_slices: Mapping[str, Tuple[int, int]], config_weights: Mapping[str, float], output_dim: int, device: torch.device) -> torch.Tensor:
    w = torch.ones(int(output_dim), dtype=torch.float32, device=device)
    for key, val in config_weights.items():
        if key in target_slices:
            a, b = target_slices[key]
            w[a:b] = float(val)
    return w


@dataclass
class TrainDiagResult:
    name: str
    status: str
    reasons: List[str]
    history: List[Dict[str, Any]]
    profile_rows: List[Dict[str, Any]]
    profile_aggregate: Dict[str, Any]
    per_target_aggregate: Dict[str, Any]
    branch_rows: List[Dict[str, Any]]
    best_epoch: int
    best_train_loss: float
    model_path: str


def _aggregate_per_target(profile_rows: Sequence[Mapping[str, Any]], target_slices: Mapping[str, Tuple[int, int]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    r2_vals: List[float] = []
    for key in target_slices.keys():
        vals = []
        maes = []
        for r in profile_rows:
            for suffix, coll in [("r2", vals), ("mae", maes)]:
                try:
                    v = float(r.get(f"{key}_{suffix}"))
                    if math.isfinite(v):
                        coll.append(v)
                except Exception:
                    pass
        if vals:
            out[f"{key}_r2_mean"] = float(np.mean(vals))
            out[f"{key}_r2_min"] = float(np.min(vals))
            out[f"{key}_r2_max"] = float(np.max(vals))
            r2_vals.extend(vals)
        if maes:
            out[f"{key}_mae_mean"] = float(np.mean(maes))
            out[f"{key}_mae_max"] = float(np.max(maes))
    out["all_target_profile_r2_mean"] = float(np.mean(r2_vals)) if r2_vals else float("nan")
    out["all_target_profile_r2_min"] = float(np.min(r2_vals)) if r2_vals else float("nan")
    return out


def _branch_aggregate_rows(profile_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    branches = sorted({str(r.get("semantic_branch", "UNKNOWN")) for r in profile_rows})
    rows: List[Dict[str, Any]] = []
    for br in branches:
        br_rows = [r for r in profile_rows if str(r.get("semantic_branch", "UNKNOWN")) == br]
        agg = aggregate_profile_rows(br_rows)
        row: Dict[str, Any] = {"semantic_branch": br, "profile_count": len(br_rows)}
        row.update(agg)
        rows.append(row)
    return rows


def train_overfit_case(
    dataset: G1Dataset,
    profiles: Sequence[ProfilePack],
    out_dir: str | Path,
    name: str,
    device_arg: str,
    epochs: int,
    lr: float,
    batch_size: int,
    model_cfg: Mapping[str, Any],
    target_group_weights: Mapping[str, float],
    eval_every: int = 25,
    pass_r2_mean_threshold: float = 0.995,
    pass_r2_min_threshold: float = 0.98,
    early_stop_on_pass: bool = True,
    min_epochs_before_early_stop: int = 100,
) -> TrainDiagResult:
    out = ensure_dir(out_dir) / name
    ensure_dir(out)
    device = _device_from_arg(device_arg)
    X = np.concatenate([p.features for p in profiles], axis=0).astype(np.float32)
    Y = np.concatenate([p.targets for p in profiles], axis=0).astype(np.float32)
    x_mean = np.nanmean(X, axis=0).astype(np.float32)
    x_std = np.nanstd(X, axis=0).astype(np.float32)
    x_std[~np.isfinite(x_std) | (x_std < 1e-8)] = 1.0
    y_mean = np.nanmean(Y, axis=0).astype(np.float32)
    y_std = np.nanstd(Y, axis=0).astype(np.float32)
    y_std[~np.isfinite(y_std) | (y_std < 1e-8)] = 1.0
    Xn = ((X - x_mean[None, :]) / x_std[None, :]).astype(np.float32)
    Yn = ((Y - y_mean[None, :]) / y_std[None, :]).astype(np.float32)

    seed = int(model_cfg.get("seed", 20260615))
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    model = GeneratorSurrogateMLP(
        input_dim=Xn.shape[1],
        output_dim=Yn.shape[1],
        width=int(model_cfg.get("width", 512)),
        depth=int(model_cfg.get("depth", 6)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    ds = TensorDataset(torch.as_tensor(Xn, dtype=torch.float32), torch.as_tensor(Yn, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(model_cfg.get("weight_decay", 1e-6)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(epochs)), eta_min=float(model_cfg.get("min_lr", 1e-5)))
    gw = make_group_weights(dataset.target_slices, target_group_weights, Yn.shape[1], device)
    history: List[Dict[str, Any]] = []
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    for ep in range(1, int(epochs) + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = (((pred - yb) ** 2) * gw[None, :]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(model_cfg.get("grad_clip_norm", 5.0)))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        if math.isfinite(avg_loss) and avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = ep
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        row: Dict[str, Any] = {"epoch": ep, "train_loss": avg_loss, "lr": float(opt.param_groups[0]["lr"])}
        if ep == 1 or ep == int(epochs) or ep % int(eval_every) == 0:
            pred = _predict_np(model, Xn, y_mean, y_std, device)
            gm = group_metrics(Y, pred, dataset.target_slices)
            row["r2_mean"] = gm["__aggregate__"]["r2_mean"]
            row["r2_min"] = gm["__aggregate__"]["r2_min"]
            try:
                if (early_stop_on_pass and ep >= int(min_epochs_before_early_stop)
                    and float(row["r2_mean"]) >= float(pass_r2_mean_threshold)
                    and float(row["r2_min"]) >= float(pass_r2_min_threshold)):
                    row["early_stop_reason"] = "PASS_THRESHOLDS_REACHED"
                    history.append(row)
                    break
            except Exception:
                pass
        history.append(row)
    if best_state is not None:
        model.load_state_dict(best_state)
    pred_all = _predict_np(model, Xn, y_mean, y_std, device)
    pred_profiles = _split_profile_predictions(pred_all, profiles)
    prof_rows = profile_metrics(profiles, pred_profiles)["rows"]
    prof_agg = aggregate_profile_rows(prof_rows)
    target_agg = _aggregate_per_target(prof_rows, dataset.target_slices)
    branch_rows = _branch_aggregate_rows(prof_rows)
    model_dir = ensure_dir(out / "model")
    model_path = model_dir / "best_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "target_slices": dataset.target_slices,
        "feature_names": dataset.feature_names,
        "target_names": dataset.target_names,
        "best_epoch": best_epoch,
        "best_train_loss": best_loss,
    }, model_path)
    write_csv(history, out / f"D17_G11_{name}_training_history.csv")
    write_csv(prof_rows, out / f"D17_G11_{name}_PROFILE_METRICS.csv")
    write_csv(branch_rows, out / f"D17_G11_{name}_BRANCH_METRICS.csv")
    reasons: List[str] = []
    mean_r2 = float(target_agg.get("all_target_profile_r2_mean", float("nan")))
    min_r2 = float(target_agg.get("all_target_profile_r2_min", float("nan")))
    if not math.isfinite(mean_r2) or mean_r2 < pass_r2_mean_threshold:
        reasons.append(f"{name} mean R2 below {pass_r2_mean_threshold}: {mean_r2:.6g}")
    if not math.isfinite(min_r2) or min_r2 < pass_r2_min_threshold:
        reasons.append(f"{name} min target/profile R2 below {pass_r2_min_threshold}: {min_r2:.6g}")
    status = "PASS" if not reasons else "REVIEW"
    return TrainDiagResult(name, status, reasons, history, prof_rows, prof_agg, target_agg, branch_rows, best_epoch, best_loss, str(model_path))


def profile_rows_to_per_target_table(rows: Sequence[Mapping[str, Any]], target_slices: Mapping[str, Tuple[int, int]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        base = {k: r.get(k, "") for k in ["split", "canonical_cell_uid", "protocol", "semantic_branch", "n_points", "r2_mean", "r2_min"]}
        for target in target_slices.keys():
            rr = dict(base)
            rr["target"] = target
            for metric in ["r2", "mae", "rmse", "corr"]:
                rr[metric] = r.get(f"{target}_{metric}", "")
            out.append(rr)
    return out


def run_g11_diagnostic(
    *,
    project_root: str | Path,
    split_manifest: str | Path,
    g0_profile_semantics_csv: str | Path,
    out_dir: str | Path,
    config: Mapping[str, Any],
    single_profile_count: int,
    train_profile_count: int,
    validation_profile_count: int,
    max_time_points: int,
    time_window_s: float,
    device_arg: str,
) -> Dict[str, Any]:
    out = ensure_dir(out_dir)
    scan = scan_generator_code(project_root)
    # Dataset used for selected train + validation diagnostics.
    ds = build_g1_dataset(
        split_manifest=split_manifest,
        g0_profile_semantics_csv=g0_profile_semantics_csv,
        train_profile_count=int(max(train_profile_count, single_profile_count, 1)),
        validation_profile_count=int(max(validation_profile_count, 0)),
        max_time_points=int(max_time_points),
        time_window_s=float(time_window_s),
    )
    selected_for_audit = list(ds.train_profiles) + list(ds.validation_profiles)
    target_stats, feature_stats = _feature_target_stats(ds, selected_for_audit)
    time_rows = _profile_time_alignment_rows(selected_for_audit)
    consistency_rows = [_estimate_csmax_consistency(p) for p in selected_for_audit]
    write_csv(target_stats, out / "D17_G11_TARGET_NORMALIZATION_AUDIT.csv")
    write_csv(feature_stats, out / "D17_G11_FEATURE_NORMALIZATION_AUDIT.csv")
    write_csv(time_rows, out / "D17_G11_TIME_GRID_ALIGNMENT_AUDIT.csv")
    write_csv(consistency_rows, out / "D17_G11_CS_THETA_CONSISTENCY_AUDIT.csv")
    json_dump(scan, out / "D17_G11_GENERATOR_CODE_SCAN.json")

    cfg_single = dict(config.get("single_profile_overfit", {}))
    cfg_closed = dict(config.get("closedset_train", {}))
    target_weights = dict(config.get("target_group_weights", {}))
    single_profiles = ds.train_profiles[: int(single_profile_count)]
    if not single_profiles:
        raise ValueError("No single-profile diagnostic profile selected")
    single = train_overfit_case(
        ds,
        single_profiles,
        out / "runs",
        name="single_profile_overfit",
        device_arg=device_arg,
        epochs=int(cfg_single.get("epochs", 800)),
        lr=float(cfg_single.get("lr", 1e-3)),
        batch_size=int(cfg_single.get("batch_size", 512)),
        model_cfg=dict(cfg_single.get("model", {})),
        target_group_weights=target_weights,
        eval_every=int(cfg_single.get("eval_every", 50)),
        pass_r2_mean_threshold=float(cfg_single.get("pass_r2_mean_threshold", 0.995)),
        pass_r2_min_threshold=float(cfg_single.get("pass_r2_min_threshold", 0.98)),
        early_stop_on_pass=bool(cfg_single.get("early_stop_on_pass", True)),
        min_epochs_before_early_stop=int(cfg_single.get("min_epochs_before_early_stop", 100)),
    )
    closed_profiles = ds.train_profiles[: int(train_profile_count)]
    closed = train_overfit_case(
        ds,
        closed_profiles,
        out / "runs",
        name="train_closedset_12profile",
        device_arg=device_arg,
        epochs=int(cfg_closed.get("epochs", 600)),
        lr=float(cfg_closed.get("lr", 8e-4)),
        batch_size=int(cfg_closed.get("batch_size", 1024)),
        model_cfg=dict(cfg_closed.get("model", {})),
        target_group_weights=target_weights,
        eval_every=int(cfg_closed.get("eval_every", 50)),
        pass_r2_mean_threshold=float(cfg_closed.get("pass_r2_mean_threshold", 0.98)),
        pass_r2_min_threshold=float(cfg_closed.get("pass_r2_min_threshold", 0.95)),
        early_stop_on_pass=bool(cfg_closed.get("early_stop_on_pass", True)),
        min_epochs_before_early_stop=int(cfg_closed.get("min_epochs_before_early_stop", 150)),
    )
    write_csv(profile_rows_to_per_target_table(single.profile_rows, ds.target_slices), out / "D17_G11_SINGLE_PROFILE_PER_TARGET_METRICS.csv")
    write_csv(profile_rows_to_per_target_table(closed.profile_rows, ds.target_slices), out / "D17_G11_CLOSEDSET_PER_TARGET_METRICS.csv")

    reasons: List[str] = []
    if scan["status"] != "PASS":
        reasons.append("generator code scan REVIEW; local generator files/patterns not fully available")
    if single.status != "PASS":
        reasons.extend(single.reasons)
    if closed.status != "PASS":
        reasons.extend(closed.reasons)
    status = "PASS" if not reasons else "REVIEW"
    if single.status != "PASS":
        recommendation = "FIX_LOADER_NORMALIZATION_OR_MODEL_BEFORE_G1"
    elif closed.status != "PASS":
        recommendation = "FIX_CAPACITY_BRANCH_HEAD_OR_TARGET_SCALING_BEFORE_G2"
    else:
        recommendation = "G1_CORE_CAN_OVERFIT_TRAIN_DATA_RERUN_G1_WITH_STRONGER_CONFIG"
    summary: Dict[str, Any] = {
        "protocol": "D17-G1.1_CLOSEDSET_ALIGNMENT_DIAGNOSTIC",
        "created_at_utc": utc_now(),
        "status": status,
        "reasons": reasons,
        "recommendation": recommendation,
        "purpose": "Diagnose whether G1 low train/validation R2 is caused by loader/time-grid/normalization/model capacity before any G2 expansion.",
        "policy": {
            "train_cell_softlabels_used_for_diagnostic_training": True,
            "validation_softlabels_report_only": True,
            "frozen_test_softlabels_used": False,
            "checkpoint_selection": "diagnostic train loss only",
            "not_a_promotion_run": True,
        },
        "generator_reference": {
            "g0_profile_semantics_csv": str(g0_profile_semantics_csv),
            "semantic_branch_counts_expected_from_G0": "Use D17_G0_PROFILE_SEMANTICS.csv; G0 should have PASS/g1_ready=true before this diagnostic.",
            "local_generator_code_scan_status": scan["status"],
            "local_generator_code_scan_file": str(out / "D17_G11_GENERATOR_CODE_SCAN.json"),
        },
        "dataset": ds.manifest_summary,
        "single_profile_overfit": {
            "status": single.status,
            "reasons": single.reasons,
            "best_epoch": single.best_epoch,
            "best_train_loss": single.best_train_loss,
            "profile_aggregate": single.profile_aggregate,
            "per_target_aggregate": single.per_target_aggregate,
            "branch_metrics": single.branch_rows,
            "model_path": single.model_path,
        },
        "train_closedset_12profile": {
            "status": closed.status,
            "reasons": closed.reasons,
            "best_epoch": closed.best_epoch,
            "best_train_loss": closed.best_train_loss,
            "profile_aggregate": closed.profile_aggregate,
            "per_target_aggregate": closed.per_target_aggregate,
            "branch_metrics": closed.branch_rows,
            "model_path": closed.model_path,
        },
        "files": {
            "summary_json": str(out / "D17_G11_CLOSEDSET_ALIGNMENT_DIAGNOSTIC_SUMMARY.json"),
            "generator_code_scan_json": str(out / "D17_G11_GENERATOR_CODE_SCAN.json"),
            "target_normalization_audit_csv": str(out / "D17_G11_TARGET_NORMALIZATION_AUDIT.csv"),
            "feature_normalization_audit_csv": str(out / "D17_G11_FEATURE_NORMALIZATION_AUDIT.csv"),
            "time_grid_alignment_audit_csv": str(out / "D17_G11_TIME_GRID_ALIGNMENT_AUDIT.csv"),
            "cs_theta_consistency_audit_csv": str(out / "D17_G11_CS_THETA_CONSISTENCY_AUDIT.csv"),
            "single_profile_per_target_metrics_csv": str(out / "D17_G11_SINGLE_PROFILE_PER_TARGET_METRICS.csv"),
            "closedset_per_target_metrics_csv": str(out / "D17_G11_CLOSEDSET_PER_TARGET_METRICS.csv"),
        },
    }
    json_dump(summary, out / "D17_G11_CLOSEDSET_ALIGNMENT_DIAGNOSTIC_SUMMARY.json")
    return summary
