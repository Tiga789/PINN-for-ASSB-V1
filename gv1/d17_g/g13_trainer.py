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

from .g1_data import G1Dataset, ProfilePack, build_g1_dataset, json_dump, save_profile_predictions
from .g1_metrics import aggregate_profile_rows, group_metrics, profile_metrics, r2_score
from .g13_model import ObservedProfileConditionedMultiHeadSurrogate

OBS_TIME_KEYS = ["t_global_s", "time_s", "t_s", "t", "time"]
OBS_I_KEYS = ["I_profile", "current_A", "I_A", "current", "I"]
OBS_V_KEYS = ["voltage_exp", "voltage_V", "V_exp", "V", "voltage"]
OBS_T_KEYS = ["temperature_C", "temp_C", "T_C", "temperature_K", "T", "temperature"]


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _device_from_arg(arg: str) -> torch.device:
    if str(arg) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(arg))


def _write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fields.append(k)
                seen.add(k)
    if not fields:
        fields = ["empty"]
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))


def _load_npz_selected(path: str | Path, keys: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    p = Path(path)
    if (not p.exists()) or p.is_dir():
        return {}
    out: Dict[str, Any] = {}
    with np.load(p, allow_pickle=True) as z:
        use = list(z.files) if keys is None else [k for k in keys if k in z.files]
        for k in use:
            out[k] = z[k]
    return out


def _to_1d_float(x: Any) -> Optional[np.ndarray]:
    try:
        a = np.asarray(x)
        if a.dtype.kind in {"U", "S", "O"}:
            return None
        y = a.astype(np.float32).reshape(-1)
        return y if y.size else None
    except Exception:
        return None


def _find_any_1d(d: Mapping[str, Any], keys: Sequence[str]) -> Tuple[Optional[str], Optional[np.ndarray]]:
    for k in keys:
        if k not in d:
            continue
        arr = _to_1d_float(d[k])
        if arr is not None and arr.size > 0:
            return k, arr
    return None, None


def _interp(src_y: Optional[np.ndarray], src_t: Optional[np.ndarray], target_t: np.ndarray, fill: float) -> np.ndarray:
    target_t = np.asarray(target_t, dtype=np.float32).reshape(-1)
    if src_y is None or src_y.size == 0:
        return np.full(target_t.size, float(fill), dtype=np.float32)
    src_y = np.asarray(src_y, dtype=np.float32).reshape(-1)
    if src_y.size == target_t.size:
        return src_y.astype(np.float32)
    if src_t is None or np.asarray(src_t).reshape(-1).size != src_y.size:
        x_old = np.linspace(0.0, 1.0, src_y.size, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, target_t.size, dtype=np.float32)
        return np.interp(x_new, x_old, src_y).astype(np.float32)
    src_t = np.asarray(src_t, dtype=np.float32).reshape(-1)
    good = np.isfinite(src_t) & np.isfinite(src_y)
    if not np.any(good):
        return np.full(target_t.size, float(fill), dtype=np.float32)
    x = src_t[good]
    y = src_y[good]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    ux, idx = np.unique(x, return_index=True)
    y = y[idx]
    if ux.size <= 1:
        return np.full(target_t.size, float(y[0]) if y.size else float(fill), dtype=np.float32)
    return np.interp(target_t, ux, y, left=float(y[0]), right=float(y[-1])).astype(np.float32)


def _replay_observed_aligned(profile: ProfilePack) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    keys = list(set(OBS_TIME_KEYS + OBS_I_KEYS + OBS_V_KEYS + OBS_T_KEYS))
    replay = _load_npz_selected(profile.replay_npz, keys)
    tk, rt = _find_any_1d(replay, OBS_TIME_KEYS)
    ik, ri = _find_any_1d(replay, OBS_I_KEYS)
    vk, rv = _find_any_1d(replay, OBS_V_KEYS)
    kk, rtemp = _find_any_1d(replay, OBS_T_KEYS)
    t = np.asarray(profile.t_global_s, dtype=np.float32).reshape(-1)
    # No soft-label target fallback for voltage/current: these are observed
    # inputs and should come from replay.  If missing, use neutral fills and
    # record the issue in the feature audit.
    I = _interp(ri, rt if rt is not None and ri is not None and rt.size == ri.size else None, t, fill=0.0)
    V = _interp(rv, rt if rt is not None and rv is not None and rt.size == rv.size else None, t, fill=0.0)
    T = _interp(rtemp, rt if rt is not None and rtemp is not None and rt.size == rtemp.size else None, t, fill=25.0)
    info = {
        "replay_time_key": tk or "missing",
        "I_key": ik or "missing_filled_zero",
        "V_key": vk or "missing_filled_zero",
        "T_key": kk or "missing_filled_25C",
        "replay_npz_exists": bool(Path(profile.replay_npz).exists()),
    }
    return I.astype(np.float32), V.astype(np.float32), T.astype(np.float32), info


def _cum_charge_ah(t: np.ndarray, I: np.ndarray) -> np.ndarray:
    if t.size == 0:
        return np.zeros_like(I, dtype=np.float32)
    dt = np.diff(t, prepend=t[0]).astype(np.float32)
    return (np.cumsum(I.astype(np.float32) * dt) / 3600.0).astype(np.float32)


def _safe_stat(x: np.ndarray, fn: str, default: float = 0.0) -> float:
    try:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float(default)
        if fn == "mean":
            return float(np.mean(arr))
        if fn == "std":
            return float(np.std(arr))
        if fn == "min":
            return float(np.min(arr))
        if fn == "max":
            return float(np.max(arr))
        if fn.startswith("q"):
            return float(np.quantile(arr, float(fn[1:]) / 100.0))
    except Exception:
        return float(default)
    return float(default)


def _profile_summary_features(profile: ProfilePack, I: np.ndarray, V: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    t = np.asarray(profile.t_global_s, dtype=np.float32).reshape(-1)
    q = _cum_charge_ah(t, I)
    dV = np.diff(V, prepend=V[0]).astype(np.float32) if V.size else np.zeros_like(I, dtype=np.float32)
    dt = np.diff(t, prepend=t[0]).astype(np.float32) if t.size else np.ones_like(I, dtype=np.float32)
    dvdt = dV / np.maximum(np.abs(dt), 1e-6)
    eps = max(1e-8, 0.001 * float(np.nanmax(np.abs(I)) + 1e-12)) if I.size else 1e-8
    charge_frac = float(np.mean(I > eps)) if I.size else 0.0
    discharge_frac = float(np.mean(I < -eps)) if I.size else 0.0
    rest_frac = float(np.mean(np.abs(I) <= eps)) if I.size else 0.0
    charge_Ah = float(np.sum(np.maximum(I, 0.0) * np.diff(t, prepend=t[0]) / 3600.0)) if I.size and t.size else 0.0
    discharge_Ah = float(np.sum(np.maximum(-I, 0.0) * np.diff(t, prepend=t[0]) / 3600.0)) if I.size and t.size else 0.0
    duration_s = float(t[-1] - t[0]) if t.size > 1 else 0.0
    vals = [
        duration_s,
        _safe_stat(V, "mean"), _safe_stat(V, "std"), _safe_stat(V, "min"), _safe_stat(V, "max"),
        float(V[0]) if V.size else 0.0, float(V[-1]) if V.size else 0.0,
        _safe_stat(V, "q05"), _safe_stat(V, "q50"), _safe_stat(V, "q95"),
        _safe_stat(I, "mean"), _safe_stat(I, "std"), _safe_stat(I, "min"), _safe_stat(I, "max"),
        _safe_stat(np.abs(I), "mean"), _safe_stat(np.abs(I), "max"),
        charge_Ah, discharge_Ah, float(q[-1]) if q.size else 0.0, _safe_stat(q, "min"), _safe_stat(q, "max"),
        rest_frac, charge_frac, discharge_frac,
        _safe_stat(dvdt[: max(2, min(64, dvdt.size))], "mean") if dvdt.size else 0.0,
        _safe_stat(dvdt[-max(2, min(64, dvdt.size)):], "mean") if dvdt.size else 0.0,
        _safe_stat(T, "mean"), _safe_stat(T, "std"), _safe_stat(T, "min"), _safe_stat(T, "max"),
    ]
    names = [
        "profile_duration_s",
        "profile_V_mean", "profile_V_std", "profile_V_min", "profile_V_max",
        "profile_V_start", "profile_V_end", "profile_V_q05", "profile_V_q50", "profile_V_q95",
        "profile_I_mean", "profile_I_std", "profile_I_min", "profile_I_max",
        "profile_absI_mean", "profile_absI_max",
        "profile_charge_Ah", "profile_discharge_Ah", "profile_net_Ah", "profile_q_min", "profile_q_max",
        "profile_rest_frac", "profile_charge_frac", "profile_discharge_frac",
        "profile_early_dVdt_mean", "profile_late_dVdt_mean",
        "profile_T_mean", "profile_T_std", "profile_T_min", "profile_T_max",
    ]
    return np.asarray(vals, dtype=np.float32), names


def _local_observed_features(profile: ProfilePack, I: np.ndarray, V: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    t = np.asarray(profile.t_global_s, dtype=np.float32).reshape(-1)
    q = _cum_charge_ah(t, I)
    dV = np.diff(V, prepend=V[0]).astype(np.float32) if V.size else np.zeros_like(I, dtype=np.float32)
    dI = np.diff(I, prepend=I[0]).astype(np.float32) if I.size else np.zeros_like(I, dtype=np.float32)
    X = np.stack([I, np.abs(I), dI, V, dV, T, q], axis=1).astype(np.float32)
    names = ["I_A_abs", "absI_A_abs", "dI_A_abs", "voltage_exp_V_abs", "dV_exp_V_abs", "temperature_C_abs", "q_Ah_abs"]
    return X, names


@dataclass
class G13PreparedData:
    base: G1Dataset
    fit_profiles: List[ProfilePack]
    internal_profiles: List[ProfilePack]
    validation_profiles: List[ProfilePack]
    X_fit: np.ndarray
    Y_fit: np.ndarray
    X_train_all: np.ndarray
    Y_train_all: np.ndarray
    X_internal: np.ndarray
    Y_internal: np.ndarray
    X_validation: np.ndarray
    Y_validation: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    feature_names: List[str]
    local_input_dim: int
    profile_input_dim: int
    profile_feature_names: List[str]
    feature_audit_rows: List[Dict[str, Any]]
    per_profile_X: Dict[str, np.ndarray]


def prepare_g13_data(base: G1Dataset, internal_heldout_count: int = 2) -> G13PreparedData:
    n_train = len(base.train_profiles)
    ih = int(max(0, min(internal_heldout_count, max(0, n_train - 1))))
    fit_profiles = base.train_profiles[:-ih] if ih > 0 else list(base.train_profiles)
    internal_profiles = base.train_profiles[-ih:] if ih > 0 else []
    local_dim = len(base.feature_names)
    feature_audit_rows: List[Dict[str, Any]] = []
    per_profile_X: Dict[str, np.ndarray] = {}
    profile_feature_names: Optional[List[str]] = None
    feature_names: Optional[List[str]] = None

    def aug(profile: ProfilePack) -> np.ndarray:
        nonlocal profile_feature_names, feature_names
        I, V, T, info = _replay_observed_aligned(profile)
        local_obs, local_obs_names = _local_observed_features(profile, I, V, T)
        pfeat, pnames = _profile_summary_features(profile, I, V, T)
        if profile_feature_names is None:
            profile_feature_names = pnames
        prepeated = np.repeat(pfeat.reshape(1, -1), profile.features.shape[0], axis=0).astype(np.float32)
        X = np.concatenate([profile.features.astype(np.float32), local_obs, prepeated], axis=1).astype(np.float32)
        if feature_names is None:
            feature_names = list(profile.feature_names) + local_obs_names + pnames
        feature_audit_rows.append({
            "canonical_cell_uid": profile.canonical_cell_uid,
            "split": profile.split,
            "protocol": profile.protocol,
            "semantic_branch": profile.branch,
            **info,
            "n_time": int(profile.features.shape[0]),
            "V_mean_abs": float(np.nanmean(V)) if V.size else float("nan"),
            "I_abs_max": float(np.nanmax(np.abs(I))) if I.size else float("nan"),
        })
        per_profile_X[profile.canonical_cell_uid] = X
        return X

    all_train_X = [aug(p) for p in base.train_profiles]
    val_X = [aug(p) for p in base.validation_profiles]
    fit_X = [per_profile_X[p.canonical_cell_uid] for p in fit_profiles]
    internal_X = [per_profile_X[p.canonical_cell_uid] for p in internal_profiles]

    X_fit = np.concatenate(fit_X, axis=0).astype(np.float32)
    Y_fit = np.concatenate([p.targets for p in fit_profiles], axis=0).astype(np.float32)
    X_all = np.concatenate(all_train_X, axis=0).astype(np.float32)
    Y_all = np.concatenate([p.targets for p in base.train_profiles], axis=0).astype(np.float32)
    X_int = np.concatenate(internal_X, axis=0).astype(np.float32) if internal_X else np.zeros((0, X_fit.shape[1]), dtype=np.float32)
    Y_int = np.concatenate([p.targets for p in internal_profiles], axis=0).astype(np.float32) if internal_profiles else np.zeros((0, Y_fit.shape[1]), dtype=np.float32)
    X_val = np.concatenate(val_X, axis=0).astype(np.float32) if val_X else np.zeros((0, X_fit.shape[1]), dtype=np.float32)
    Y_val = np.concatenate([p.targets for p in base.validation_profiles], axis=0).astype(np.float32) if base.validation_profiles else np.zeros((0, Y_fit.shape[1]), dtype=np.float32)

    x_mean = np.nanmean(X_fit, axis=0).astype(np.float32)
    x_std = np.nanstd(X_fit, axis=0).astype(np.float32)
    x_std[~np.isfinite(x_std) | (x_std < 1e-8)] = 1.0
    y_mean = np.nanmean(Y_fit, axis=0).astype(np.float32)
    y_std = np.nanstd(Y_fit, axis=0).astype(np.float32)
    y_std[~np.isfinite(y_std) | (y_std < 1e-8)] = 1.0

    pf_names = profile_feature_names or []
    names = feature_names or []
    return G13PreparedData(
        base=base,
        fit_profiles=fit_profiles,
        internal_profiles=internal_profiles,
        validation_profiles=base.validation_profiles,
        X_fit=X_fit,
        Y_fit=Y_fit,
        X_train_all=X_all,
        Y_train_all=Y_all,
        X_internal=X_int,
        Y_internal=Y_int,
        X_validation=X_val,
        Y_validation=Y_val,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        feature_names=names,
        local_input_dim=len(base.feature_names) + 7,
        profile_input_dim=len(pf_names),
        profile_feature_names=pf_names,
        feature_audit_rows=feature_audit_rows,
        per_profile_X=per_profile_X,
    )


def _norm_X(X: np.ndarray, data: G13PreparedData) -> np.ndarray:
    out = ((X - data.x_mean[None, :]) / data.x_std[None, :]).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def _norm_Y(Y: np.ndarray, data: G13PreparedData) -> np.ndarray:
    out = ((Y - data.y_mean[None, :]) / data.y_std[None, :]).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def _group_balanced_loss(pred: torch.Tensor, target: torch.Tensor, target_slices: Mapping[str, Tuple[int, int]], weights: Mapping[str, float]) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    wsum = 0.0
    for key, (a, b) in target_slices.items():
        weight = float(weights.get(key, 1.0))
        if weight <= 0:
            continue
        losses.append(torch.mean((pred[:, a:b] - target[:, a:b]) ** 2) * weight)
        wsum += weight
    if not losses:
        return torch.mean((pred - target) ** 2)
    return torch.stack(losses).sum() / max(wsum, 1e-12)


def _predict_np(model: torch.nn.Module, X: np.ndarray, data: G13PreparedData, device: torch.device, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    outs: List[np.ndarray] = []
    Xn = _norm_X(X, data)
    with torch.no_grad():
        for i in range(0, Xn.shape[0], batch_size):
            xb = torch.as_tensor(Xn[i : i + batch_size], dtype=torch.float32, device=device)
            yp = model(xb).detach().cpu().numpy()
            outs.append(yp)
    yn = np.concatenate(outs, axis=0) if outs else np.zeros((0, data.y_mean.size), dtype=np.float32)
    return (yn * data.y_std[None, :] + data.y_mean[None, :]).astype(np.float32)


def _predict_profiles(model: torch.nn.Module, profiles: Sequence[ProfilePack], data: G13PreparedData, device: torch.device) -> List[np.ndarray]:
    return [_predict_np(model, data.per_profile_X[p.canonical_cell_uid], data, device) for p in profiles]


def _per_target_rows(profiles: Sequence[ProfilePack], preds: Sequence[np.ndarray], split_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for prof, pred in zip(profiles, preds):
        for key, (a, b) in prof.target_slices.items():
            yt = prof.targets[:, a:b]
            yp = pred[:, a:b]
            rows.append({
                "split": split_name,
                "canonical_cell_uid": prof.canonical_cell_uid,
                "protocol": prof.protocol,
                "semantic_branch": prof.branch,
                "target": key,
                "mae": float(np.nanmean(np.abs(yp - yt))),
                "rmse": float(np.sqrt(np.nanmean((yp - yt) ** 2))),
                "r2": r2_score(yt, yp),
                "n_points": int(yt.size),
            })
    return rows


def _target_aggregate(rows: Sequence[Mapping[str, Any]], split_name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"split": split_name}
    vals: List[float] = []
    targets = sorted({str(r.get("target")) for r in rows})
    for t in targets:
        rs = []
        for r in rows:
            if str(r.get("target")) != t:
                continue
            try:
                v = float(r.get("r2", float("nan")))
            except Exception:
                v = float("nan")
            if np.isfinite(v):
                rs.append(v)
        if rs:
            out[f"{t}_r2_mean"] = float(np.mean(rs))
            out[f"{t}_r2_min"] = float(np.min(rs))
            vals.extend(rs)
    out["all_target_profile_r2_mean"] = float(np.mean(vals)) if vals else float("nan")
    out["all_target_profile_r2_min"] = float(np.min(vals)) if vals else float("nan")
    return out


def _normalization_audit(data: G13PreparedData) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, (a, b) in data.base.target_slices.items():
        arr = data.Y_fit[:, a:b]
        rows.append({
            "target": key,
            "dim": int(b - a),
            "mean_abs": float(np.nanmean(np.abs(arr))),
            "std_mean": float(np.nanmean(np.nanstd(arr, axis=0))),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "y_std_min": float(np.nanmin(data.y_std[a:b])),
            "y_std_max": float(np.nanmax(data.y_std[a:b])),
        })
    return rows


def _aggregate_or_empty(profiles: Sequence[ProfilePack], preds: Sequence[np.ndarray], split_name: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    if not profiles:
        return [], [], {"split": split_name}, {"split": split_name}
    rows = profile_metrics(profiles, preds)["rows"]
    trows = _per_target_rows(profiles, preds, split_name)
    return rows, trows, aggregate_profile_rows(rows), _target_aggregate(trows, split_name)


def train_g13_validation_aware(
    base: G1Dataset,
    out_dir: str | Path,
    config: Mapping[str, Any],
    device_arg: str = "auto",
    epochs: int = 700,
    lr: float = 8e-4,
    batch_size: int = 1024,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = _device_from_arg(device_arg)
    seed = int(config.get("seed", 20260615))
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.set_num_threads(int(config.get("torch_num_threads", 2)))
    except Exception:
        pass

    internal_count = int(config.get("internal_heldout_profile_count", 2))
    data = prepare_g13_data(base, internal_heldout_count=internal_count)
    Xn = _norm_X(data.X_fit, data)
    Yn = _norm_Y(data.Y_fit, data)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(Xn), torch.as_tensor(Yn)),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )

    model_cfg = dict(config.get("model", {}))
    model = ObservedProfileConditionedMultiHeadSurrogate(
        local_input_dim=data.local_input_dim,
        profile_input_dim=data.profile_input_dim,
        target_slices=data.base.target_slices,
        width=int(model_cfg.get("width", 768)),
        depth=int(model_cfg.get("depth", 7)),
        profile_width=int(model_cfg.get("profile_width", 192)),
        dropout=float(model_cfg.get("dropout", 0.03)),
        phie_direct_width=int(model_cfg.get("phie_direct_width", 192)),
    ).to(device)
    group_weights = dict(config.get("target_group_weights", {
        "theta_a": 1.5,
        "theta_c": 1.5,
        "cs_a": 1.0,
        "cs_c": 1.0,
        "phie": 10.0,
        "phis_c": 2.5,
    }))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(model_cfg.get("weight_decay", 1e-6)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(epochs)), eta_min=float(model_cfg.get("min_lr", 1e-5)))

    eval_every = int(config.get("eval_every", 50))
    min_epochs_before_stop = int(config.get("min_epochs_before_early_stop", 250))
    fit_mean_thr = float(config.get("fit_train_r2_mean_threshold", 0.99))
    fit_min_thr = float(config.get("fit_train_r2_min_threshold", 0.97))
    internal_mean_thr = float(config.get("internal_heldout_r2_mean_threshold", 0.95))
    internal_min_thr = float(config.get("internal_heldout_r2_min_threshold", 0.90))

    history: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {"epoch": 0, "score": -1e99, "fit_loss": float("inf"), "state_dict": None}

    for ep in range(1, int(epochs) + 1):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            pred = model(xb)
            loss = _group_balanced_loss(pred, yb, data.base.target_slices, group_weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(model_cfg.get("grad_clip_norm", 5.0)))
            opt.step()
            batch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        fit_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        row: Dict[str, Any] = {"epoch": ep, "fit_train_loss": fit_loss, "lr": float(opt.param_groups[0]["lr"])}
        do_eval = ep == 1 or ep == int(epochs) or ep % eval_every == 0
        if do_eval:
            pred_fit = _predict_np(model, data.X_fit, data, device)
            fit_gm = group_metrics(data.Y_fit, pred_fit, data.base.target_slices)
            row["fit_train_r2_mean"] = float(fit_gm["__aggregate__"]["r2_mean"])
            row["fit_train_r2_min"] = float(fit_gm["__aggregate__"]["r2_min"])
            if data.X_internal.shape[0] > 0:
                pred_int = _predict_np(model, data.X_internal, data, device)
                int_gm = group_metrics(data.Y_internal, pred_int, data.base.target_slices)
                row["internal_heldout_r2_mean"] = float(int_gm["__aggregate__"]["r2_mean"])
                row["internal_heldout_r2_min"] = float(int_gm["__aggregate__"]["r2_min"])
            else:
                row["internal_heldout_r2_mean"] = float("nan")
                row["internal_heldout_r2_min"] = float("nan")
            for key in ["theta_a", "theta_c", "cs_a", "cs_c", "phie", "phis_c"]:
                if key in fit_gm:
                    row[f"fit_{key}_r2"] = float(fit_gm[key]["r2"])
            score = row["fit_train_r2_mean"] + 0.2 * row["fit_train_r2_min"]
            if np.isfinite(row.get("internal_heldout_r2_mean", float("nan"))):
                score += 0.5 * row["internal_heldout_r2_mean"] + 0.1 * row["internal_heldout_r2_min"]
            score -= 0.01 * fit_loss
            if np.isfinite(score) and score > best["score"]:
                best = {"epoch": ep, "score": float(score), "fit_loss": fit_loss, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
            if bool(config.get("early_stop_on_internal_pass", False)) and ep >= min_epochs_before_stop:
                fit_ok = row["fit_train_r2_mean"] >= fit_mean_thr and row["fit_train_r2_min"] >= fit_min_thr
                int_ok = True
                if data.X_internal.shape[0] > 0:
                    int_ok = row["internal_heldout_r2_mean"] >= internal_mean_thr and row["internal_heldout_r2_min"] >= internal_min_thr
                if fit_ok and int_ok:
                    history.append(row)
                    break
        elif math.isfinite(fit_loss) and fit_loss < best["fit_loss"]:
            best = {"epoch": ep, "score": best.get("score", -1e99), "fit_loss": fit_loss, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
        history.append(row)

    if best.get("state_dict") is not None:
        model.load_state_dict(best["state_dict"])

    fit_preds = _predict_profiles(model, data.fit_profiles, data, device)
    internal_preds = _predict_profiles(model, data.internal_profiles, data, device) if data.internal_profiles else []
    all_train_preds = _predict_profiles(model, data.base.train_profiles, data, device)
    validation_preds = _predict_profiles(model, data.validation_profiles, data, device) if data.validation_profiles else []

    fit_rows, fit_trows, fit_agg, fit_tagg = _aggregate_or_empty(data.fit_profiles, fit_preds, "train_fit")
    int_rows, int_trows, int_agg, int_tagg = _aggregate_or_empty(data.internal_profiles, internal_preds, "train_internal_heldout")
    all_rows, all_trows, all_agg, all_tagg = _aggregate_or_empty(data.base.train_profiles, all_train_preds, "train_all_report")
    val_rows, val_trows, val_agg, val_tagg = _aggregate_or_empty(data.validation_profiles, validation_preds, "validation_report_only")

    pred_manifest: List[Dict[str, Any]] = []
    pred_manifest.extend(save_profile_predictions(out / "predictions" / "train_fit", "G13_train_fit", data.fit_profiles, fit_preds))
    if data.internal_profiles:
        pred_manifest.extend(save_profile_predictions(out / "predictions" / "train_internal_heldout", "G13_train_internal_heldout", data.internal_profiles, internal_preds))
    if data.validation_profiles:
        pred_manifest.extend(save_profile_predictions(out / "predictions" / "validation_report_only", "G13_validation_report_only", data.validation_profiles, validation_preds))

    _write_csv(history, out / "D17_G13_training_history.csv")
    _write_csv(fit_rows + int_rows + all_rows + val_rows, out / "D17_G13_PROFILE_METRICS.csv")
    _write_csv(fit_trows + int_trows + all_trows + val_trows, out / "D17_G13_PER_TARGET_PROFILE_METRICS.csv")
    _write_csv([fit_tagg, int_tagg, all_tagg, val_tagg], out / "D17_G13_PER_TARGET_AGGREGATE.csv")
    _write_csv(_normalization_audit(data), out / "D17_G13_TARGET_NORMALIZATION_AUDIT.csv")
    _write_csv(data.feature_audit_rows, out / "D17_G13_PROFILE_ENCODER_FEATURE_AUDIT.csv")
    _write_csv(pred_manifest, out / "D17_G13_PREDICTION_MANIFEST.csv")

    model_dir = out / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "x_mean": data.x_mean,
        "x_std": data.x_std,
        "y_mean": data.y_mean,
        "y_std": data.y_std,
        "feature_names": data.feature_names,
        "target_names": data.base.target_names,
        "target_slices": data.base.target_slices,
        "local_input_dim": data.local_input_dim,
        "profile_input_dim": data.profile_input_dim,
        "profile_feature_names": data.profile_feature_names,
        "config": dict(config),
        "best_epoch": int(best.get("epoch", 0)),
        "profile_conditioning": "observed_profile_encoder_no_train_profile_id",
    }, model_dir / "best_model.pt")

    fit_mean = float(fit_tagg.get("all_target_profile_r2_mean", float("nan")))
    fit_min = float(fit_tagg.get("all_target_profile_r2_min", float("nan")))
    int_mean = float(int_tagg.get("all_target_profile_r2_mean", float("nan"))) if data.internal_profiles else float("nan")
    int_min = float(int_tagg.get("all_target_profile_r2_min", float("nan"))) if data.internal_profiles else float("nan")
    val_mean = float(val_tagg.get("all_target_profile_r2_mean", float("nan"))) if data.validation_profiles else float("nan")
    val_min = float(val_tagg.get("all_target_profile_r2_min", float("nan"))) if data.validation_profiles else float("nan")
    val_phie = float(val_tagg.get("phie_r2_mean", float("nan")))
    val_phis = float(val_tagg.get("phis_c_r2_mean", float("nan")))

    status_reasons: List[str] = []
    if not math.isfinite(fit_mean) or fit_mean < float(config.get("fit_status_r2_mean_threshold", 0.98)):
        status_reasons.append(f"fit-train mean R2 below status threshold: {fit_mean:.6g}")
    if not math.isfinite(fit_min) or fit_min < float(config.get("fit_status_r2_min_threshold", 0.95)):
        status_reasons.append(f"fit-train min R2 below status threshold: {fit_min:.6g}")
    status = "PASS" if not status_reasons else "REVIEW"

    g2_reasons: List[str] = []
    if fit_mean < fit_mean_thr or fit_min < fit_min_thr:
        g2_reasons.append(f"fit train target/profile R2 below G2 gate: mean={fit_mean:.6g}, min={fit_min:.6g}")
    if data.internal_profiles and (int_mean < internal_mean_thr or int_min < internal_min_thr):
        g2_reasons.append(f"internal heldout target/profile R2 below gate: mean={int_mean:.6g}, min={int_min:.6g}")
    val_mean_thr = float(config.get("validation_r2_mean_threshold", 0.95))
    val_min_thr = float(config.get("validation_r2_min_threshold", 0.90))
    val_phie_thr = float(config.get("validation_phie_r2_mean_threshold", 0.90))
    val_phis_thr = float(config.get("validation_phis_c_r2_mean_threshold", 0.90))
    if data.validation_profiles and (val_mean < val_mean_thr or val_min < val_min_thr):
        g2_reasons.append(f"validation report-only target/profile R2 below gate: mean={val_mean:.6g}, min={val_min:.6g}")
    if data.validation_profiles and (val_phie < val_phie_thr):
        g2_reasons.append(f"validation phie R2 below gate: mean={val_phie:.6g}")
    if data.validation_profiles and (val_phis < val_phis_thr):
        g2_reasons.append(f"validation phis_c R2 below gate: mean={val_phis:.6g}")
    g2_ready = len(g2_reasons) == 0
    recommendation = "ENTER_D17_G2_HELDOUT_SURROGATE_EXPANSION" if g2_ready else "DO_NOT_ENTER_G2_FIX_PROFILE_ENCODER_OR_TRAIN_COVERAGE"

    summary: Dict[str, Any] = {
        "protocol": "D17-G1.3_VALIDATION_AWARE_GENERATOR_SURROGATE",
        "created_at_utc": _utc_now(),
        "status": status,
        "status_reasons": status_reasons,
        "recommendation": recommendation,
        "g2_ready": bool(g2_ready),
        "g2_blockers": g2_reasons,
        "purpose": "Replace train-profile-id memorization with observed-profile-conditioned generator surrogate before any G2 expansion.",
        "policy": {
            "train_cell_softlabels_used_for_training": True,
            "validation_softlabels_report_only": True,
            "frozen_test_softlabels_used": False,
            "checkpoint_selection": "fit-train plus train-internal heldout metrics; validation report-only metrics are not used to select checkpoint",
            "not_a_G2_run": True,
        },
        "generator_semantics_used": {
            "G0_required": True,
            "D15_RG_branch_note": "RG repair branch preserves source voltage/phi labels; phie remains a dedicated head.",
            "observed_inputs_note": "Replay I(t), V(t), T(t) are aligned to the soft-label target grid and used as observed profile information; cs/theta/phie/phis targets are not used as inputs.",
        },
        "device": str(device),
        "seed": seed,
        "epochs_requested": int(epochs),
        "best_epoch": int(best.get("epoch", 0)),
        "dataset": {
            **dict(base.manifest_summary),
            "fit_train_profile_count": len(data.fit_profiles),
            "internal_heldout_profile_count": len(data.internal_profiles),
            "validation_profile_count": len(data.validation_profiles),
            "augmented_feature_dim": int(data.X_fit.shape[1]),
            "local_input_dim": int(data.local_input_dim),
            "profile_input_dim": int(data.profile_input_dim),
        },
        "model": {
            "class": "ObservedProfileConditionedMultiHeadSurrogate",
            "profile_conditioning": "observed profile summary encoder; no train profile-id embedding",
            "target_group_weights": group_weights,
        },
        "fit_train_per_target_aggregate": fit_tagg,
        "internal_heldout_per_target_aggregate": int_tagg,
        "train_all_report_per_target_aggregate": all_tagg,
        "validation_report_only_per_target_aggregate": val_tagg,
        "fit_train_profile_aggregate": fit_agg,
        "internal_heldout_profile_aggregate": int_agg,
        "validation_report_only_profile_aggregate": val_agg,
        "files": {
            "summary_json": str(out / "D17_G13_VALIDATION_AWARE_SURROGATE_SUMMARY.json"),
            "profile_metrics_csv": str(out / "D17_G13_PROFILE_METRICS.csv"),
            "per_target_profile_metrics_csv": str(out / "D17_G13_PER_TARGET_PROFILE_METRICS.csv"),
            "per_target_aggregate_csv": str(out / "D17_G13_PER_TARGET_AGGREGATE.csv"),
            "profile_encoder_feature_audit_csv": str(out / "D17_G13_PROFILE_ENCODER_FEATURE_AUDIT.csv"),
            "target_normalization_audit_csv": str(out / "D17_G13_TARGET_NORMALIZATION_AUDIT.csv"),
            "prediction_manifest_csv": str(out / "D17_G13_PREDICTION_MANIFEST.csv"),
            "training_history_csv": str(out / "D17_G13_training_history.csv"),
            "best_model_pt": str(model_dir / "best_model.pt"),
        },
    }
    json_dump(summary, out / "D17_G13_VALIDATION_AWARE_SURROGATE_SUMMARY.json")
    return summary


def build_and_train_g13(
    split_manifest: str | Path,
    g0_profile_semantics_csv: str | Path,
    out_dir: str | Path,
    config: Mapping[str, Any],
    train_profile_count: int,
    validation_profile_count: int,
    max_time_points: int,
    time_window_s: float,
    device_arg: str,
    epochs: int,
    lr: float,
    batch_size: int,
) -> Dict[str, Any]:
    ds = build_g1_dataset(
        split_manifest=split_manifest,
        g0_profile_semantics_csv=g0_profile_semantics_csv,
        train_profile_count=int(train_profile_count),
        validation_profile_count=int(validation_profile_count),
        max_time_points=int(max_time_points),
        time_window_s=float(time_window_s),
    )
    return train_g13_validation_aware(ds, out_dir, config, device_arg=device_arg, epochs=int(epochs), lr=float(lr), batch_size=int(batch_size))
