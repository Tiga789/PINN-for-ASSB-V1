from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_nn.model import build_model

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None


TIME_KEYS = ["t_global_s", "time_s", "t_s", "t", "time"]
I_KEYS = ["I_profile", "current_A", "I_A", "current", "I"]
VOLTAGE_KEYS = ["voltage_exp", "voltage_V", "V_exp", "V"]
TEMP_KEYS = ["temperature_C", "temperature_K", "temp_C", "T_C", "T"]
THETA_A_KEYS = ["theta_a", "theta_n", "theta_negative"]
THETA_C_KEYS = ["theta_c", "theta_p", "theta_positive"]
CS_A_KEYS = ["cs_a", "cs_n", "cs_negative"]
CS_C_KEYS = ["cs_c", "cs_p", "cs_positive"]
PHIE_KEYS = ["phie", "phi_e", "phi_e_eff"]
PHIS_KEYS = ["phis_c_soft", "phis_c", "voltage_soft", "V_soft", "V_pred"]


class Stat1D:
    """Streaming metrics for flattened arrays."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.n = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_err = 0.0
        self.max_abs = 0.0
        self.sum_t = 0.0
        self.sum_p = 0.0
        self.sum_t2 = 0.0
        self.sum_p2 = 0.0
        self.sum_tp = 0.0
        self.tmin = float("inf")
        self.tmax = float("-inf")
        self.pmin = float("inf")
        self.pmax = float("-inf")

    def update(self, true: np.ndarray, pred: np.ndarray) -> None:
        t = np.asarray(true, dtype=np.float64).reshape(-1)
        p = np.asarray(pred, dtype=np.float64).reshape(-1)
        m = np.isfinite(t) & np.isfinite(p)
        if not np.any(m):
            return
        t = t[m]
        p = p[m]
        e = p - t
        n = int(t.size)
        self.n += n
        ae = np.abs(e)
        self.sum_abs += float(np.sum(ae))
        self.sum_sq += float(np.sum(e * e))
        self.sum_err += float(np.sum(e))
        self.max_abs = max(self.max_abs, float(np.max(ae)))
        self.sum_t += float(np.sum(t))
        self.sum_p += float(np.sum(p))
        self.sum_t2 += float(np.sum(t * t))
        self.sum_p2 += float(np.sum(p * p))
        self.sum_tp += float(np.sum(t * p))
        self.tmin = min(self.tmin, float(np.min(t)))
        self.tmax = max(self.tmax, float(np.max(t)))
        self.pmin = min(self.pmin, float(np.min(p)))
        self.pmax = max(self.pmax, float(np.max(p)))

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {f"{self.prefix}_count": int(self.n)}
        if self.n <= 0:
            for k in ["mae", "rmse", "max_abs", "bias", "corr", "r2", "true_min", "true_max", "pred_min", "pred_max"]:
                out[f"{self.prefix}_{k}"] = None
            return out
        mae = self.sum_abs / self.n
        rmse = math.sqrt(self.sum_sq / self.n)
        bias = self.sum_err / self.n
        # Pearson corr.
        denom_t = self.sum_t2 - (self.sum_t * self.sum_t) / self.n
        denom_p = self.sum_p2 - (self.sum_p * self.sum_p) / self.n
        denom = denom_t * denom_p
        corr = None
        if denom > 1e-24:
            corr = (self.sum_tp - (self.sum_t * self.sum_p) / self.n) / math.sqrt(denom)
        # R2 vs true mean.
        ss_tot = denom_t
        r2 = None
        if ss_tot > 1e-24:
            r2 = 1.0 - self.sum_sq / ss_tot
        out.update({
            f"{self.prefix}_mae": float(mae),
            f"{self.prefix}_rmse": float(rmse),
            f"{self.prefix}_max_abs": float(self.max_abs),
            f"{self.prefix}_bias": float(bias),
            f"{self.prefix}_corr": None if corr is None or not np.isfinite(corr) else float(corr),
            f"{self.prefix}_r2": None if r2 is None or not np.isfinite(r2) else float(r2),
            f"{self.prefix}_true_min": float(self.tmin),
            f"{self.prefix}_true_max": float(self.tmax),
            f"{self.prefix}_pred_min": float(self.pmin),
            f"{self.prefix}_pred_max": float(self.pmax),
        })
        return out


class ProfileStats:
    def __init__(self, slices: Mapping[str, Tuple[int, int]]):
        self.slices = {k: (int(v[0]), int(v[1])) for k, v in slices.items()}
        self.stats = {
            "phis_c": Stat1D("phis_c"),
            "phie": Stat1D("phie"),
            "theta_a": Stat1D("theta_a"),
            "theta_c": Stat1D("theta_c"),
            "theta_a_mean": Stat1D("theta_a_mean"),
            "theta_c_mean": Stat1D("theta_c_mean"),
            "grad_a_surface_center": Stat1D("grad_a_surface_center"),
            "grad_c_surface_center": Stat1D("grad_c_surface_center"),
        }
        self.theta_total = 0
        self.theta_outside = 0
        self.theta_boundary = 0
        self.theta_pred_min = float("inf")
        self.theta_pred_max = float("-inf")
        self.theta_true_min = float("inf")
        self.theta_true_max = float("-inf")

    @staticmethod
    def _vol_weights(nr: int) -> np.ndarray:
        edges = np.linspace(0.0, 1.0, nr + 1, dtype=np.float64)
        w = edges[1:] ** 3 - edges[:-1] ** 3
        return w / np.sum(w)

    def update(self, yt: np.ndarray, yp: np.ndarray) -> None:
        s = self.slices
        ta = yt[:, s["theta_a"][0]:s["theta_a"][1]]
        pa = yp[:, s["theta_a"][0]:s["theta_a"][1]]
        tc = yt[:, s["theta_c"][0]:s["theta_c"][1]]
        pc = yp[:, s["theta_c"][0]:s["theta_c"][1]]
        phie_t = yt[:, s["phie"][0]:s["phie"][1]].reshape(-1)
        phie_p = yp[:, s["phie"][0]:s["phie"][1]].reshape(-1)
        phis_t = yt[:, s["phis_c"][0]:s["phis_c"][1]].reshape(-1)
        phis_p = yp[:, s["phis_c"][0]:s["phis_c"][1]].reshape(-1)
        self.stats["phis_c"].update(phis_t, phis_p)
        self.stats["phie"].update(phie_t, phie_p)
        self.stats["theta_a"].update(ta, pa)
        self.stats["theta_c"].update(tc, pc)
        wa = self._vol_weights(ta.shape[1])
        wc = self._vol_weights(tc.shape[1])
        self.stats["theta_a_mean"].update(np.sum(ta * wa[None, :], axis=1), np.sum(pa * wa[None, :], axis=1))
        self.stats["theta_c_mean"].update(np.sum(tc * wc[None, :], axis=1), np.sum(pc * wc[None, :], axis=1))
        self.stats["grad_a_surface_center"].update(ta[:, -1] - ta[:, 0], pa[:, -1] - pa[:, 0])
        self.stats["grad_c_surface_center"].update(tc[:, -1] - tc[:, 0], pc[:, -1] - pc[:, 0])
        all_p = np.concatenate([pa.reshape(-1), pc.reshape(-1)])
        all_t = np.concatenate([ta.reshape(-1), tc.reshape(-1)])
        m = np.isfinite(all_p)
        if np.any(m):
            p = all_p[m]
            self.theta_total += int(p.size)
            self.theta_outside += int(np.sum((p < -1e-5) | (p > 1.0 + 1e-5)))
            self.theta_boundary += int(np.sum((p <= 1e-5) | (p >= 1.0 - 1e-5)))
            self.theta_pred_min = min(self.theta_pred_min, float(np.min(p)))
            self.theta_pred_max = max(self.theta_pred_max, float(np.max(p)))
        mt = np.isfinite(all_t)
        if np.any(mt):
            t = all_t[mt]
            self.theta_true_min = min(self.theta_true_min, float(np.min(t)))
            self.theta_true_max = max(self.theta_true_max, float(np.max(t)))

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for st in self.stats.values():
            out.update(st.as_dict())
        total = max(1, self.theta_total)
        out["pred_theta_outside_fraction"] = float(self.theta_outside / total)
        out["pred_theta_boundary_hit_fraction"] = float(self.theta_boundary / total)
        out["pred_theta_min"] = None if not np.isfinite(self.theta_pred_min) else float(self.theta_pred_min)
        out["pred_theta_max"] = None if not np.isfinite(self.theta_pred_max) else float(self.theta_pred_max)
        out["true_theta_min"] = None if not np.isfinite(self.theta_true_min) else float(self.theta_true_min)
        out["true_theta_max"] = None if not np.isfinite(self.theta_true_max) else float(self.theta_true_max)
        corr_keys = ["phis_c_corr", "phie_corr", "theta_a_corr", "theta_c_corr", "theta_a_mean_corr", "theta_c_mean_corr", "grad_a_surface_center_corr", "grad_c_surface_center_corr"]
        vals = [out.get(k) for k in corr_keys if out.get(k) is not None]
        out["min_selected_corr"] = float(min(vals)) if vals else None
        return out


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))


def normalize_slices(s: Mapping[str, Any]) -> Dict[str, Tuple[int, int]]:
    return {k: (int(v[0]), int(v[1])) for k, v in s.items()}


def safe_name(profile_id: str) -> str:
    return profile_id.replace("/", "__").replace("\\", "__").replace(":", "_")


def profile_id_from_npz(npz_path: Path, root: Path) -> str:
    try:
        rel = npz_path.parent.relative_to(root)
        s = str(rel).replace("\\", "/")
        return s if s not in ("", ".") else npz_path.parent.name
    except Exception:
        return npz_path.parent.name or npz_path.stem


def discover_npz(root: Path, filename: str = "solution_softlabels.npz") -> List[Path]:
    return sorted(root.rglob(filename))


def parse_batch_protocol(pid: str) -> Tuple[str, str, str]:
    batch = "unknown"
    protocol = "unknown"
    cell = pid.replace("\\", "/").split("/")[-1]
    m = re.search(r"Batch[-_ ]?(\d+)", pid, flags=re.I)
    if m:
        batch = f"Batch-{m.group(1)}"
    # Heuristic protocol parsing from common folder names.
    for cand in ["2C", "R2.5", "R3", "random", "random_walk", "GEO", "geo", "3C"]:
        if cand.lower() in pid.lower():
            protocol = cand
            break
    if protocol == "unknown":
        if batch == "Batch-1": protocol = "2C"
        elif batch == "Batch-2": protocol = "3C"
        elif batch == "Batch-3": protocol = "R2.5"
        elif batch == "Batch-4": protocol = "R3"
        elif batch == "Batch-5": protocol = "random_walk"
        elif batch == "Batch-6": protocol = "GEO"
    return batch, protocol, cell


def choose_profile_index(pid: str, profile_ids: Sequence[str]) -> Tuple[int, str]:
    if pid in profile_ids:
        return profile_ids.index(pid), "exact"
    b, prot, _ = parse_batch_protocol(pid)
    # Prefer same batch.
    for i, sid in enumerate(profile_ids):
        sb, sp, _ = parse_batch_protocol(sid)
        if sb == b:
            return i, f"same_batch:{sid}"
    # Batch-2 maps best to Batch-1 fixed full-cycle if available.
    if b == "Batch-2":
        for i, sid in enumerate(profile_ids):
            sb, sp, _ = parse_batch_protocol(sid)
            if sb == "Batch-1":
                return i, f"batch2_to_batch1:{sid}"
    # Prefer same protocol class.
    for i, sid in enumerate(profile_ids):
        sb, sp, _ = parse_batch_protocol(sid)
        if sp == prot:
            return i, f"same_protocol:{sid}"
    return 0, f"fallback_first:{profile_ids[0] if profile_ids else 'none'}"


def find_member(zf: zipfile.ZipFile, keys: Sequence[str]) -> Optional[str]:
    names = zf.namelist()
    low = {n.lower(): n for n in names}
    for k in keys:
        cand = f"{k}.npy".lower()
        if cand in low:
            return low[cand]
    # tolerate nested paths in npz, though uncommon.
    for k in keys:
        for n in names:
            if Path(n).name.lower() == f"{k}.npy".lower():
                return n
    return None


def extract_member_to_npy(npz_path: Path, keys: Sequence[str], out_dir: Path, required: bool = True) -> Optional[Path]:
    with zipfile.ZipFile(npz_path, "r") as zf:
        member = find_member(zf, keys)
        if member is None:
            if required:
                raise KeyError(f"Missing any of {keys} in {npz_path}")
            return None
        out = out_dir / Path(member).name
        with zf.open(member, "r") as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024 * 16)
        return out


def load_mmap_npy(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r", allow_pickle=False)


def orient_get(arr: np.ndarray, start: int, end: int, n_time: int, name: str) -> np.ndarray:
    if arr.ndim == 1:
        if arr.shape[0] != n_time:
            raise ValueError(f"{name} length {arr.shape[0]} != n_time {n_time}")
        return np.asarray(arr[start:end], dtype=np.float32).reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name}: expected 1D/2D, got {arr.shape}")
    if arr.shape[0] == n_time:
        return np.asarray(arr[start:end, :], dtype=np.float32)
    if arr.shape[1] == n_time:
        return np.asarray(arr[:, start:end].T, dtype=np.float32)
    raise ValueError(f"{name}: cannot orient {arr.shape} for n_time={n_time}")


def one_d_get(arr: np.ndarray, start: int, end: int, n_time: int, name: str) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 0:
        return np.full(end - start, float(a), dtype=np.float32)
    a = a.reshape(-1)
    if a.shape[0] != n_time:
        raise ValueError(f"{name} length {a.shape[0]} != n_time {n_time}")
    return np.asarray(a[start:end], dtype=np.float32)


def streaming_mean_std(arr: np.ndarray, n_time: int, chunk: int, fill_if_bad: float = 0.0) -> Tuple[float, float]:
    s = 0.0
    s2 = 0.0
    n = 0
    for st in range(0, n_time, chunk):
        ed = min(n_time, st + chunk)
        x = one_d_get(arr, st, ed, n_time, "stat")
        m = np.isfinite(x)
        if np.any(m):
            xf = x[m].astype(np.float64)
            s += float(np.sum(xf))
            s2 += float(np.sum(xf * xf))
            n += int(xf.size)
    if n <= 0:
        return float(fill_if_bad), 1.0
    mean = s / n
    var = max(0.0, s2 / n - mean * mean)
    std = math.sqrt(var)
    if not np.isfinite(std) or std <= 1e-9:
        std = 1.0
    return float(mean), float(std)


def streaming_abs_percentile(arr: np.ndarray, n_time: int, chunk: int, percentile: float = 99.5, sample_limit: int = 500_000) -> float:
    # Deterministic strided sampling avoids reading a 30M profile fully into RAM.
    if n_time <= sample_limit:
        vals = []
        for st in range(0, n_time, chunk):
            ed = min(n_time, st + chunk)
            vals.append(np.abs(one_d_get(arr, st, ed, n_time, "I")))
        x = np.concatenate(vals) if vals else np.array([], dtype=np.float32)
    else:
        step = max(1, n_time // sample_limit)
        idx = np.arange(0, n_time, step, dtype=np.int64)[:sample_limit]
        # Indexing a memmap with a huge fancy index can allocate; do it in pieces.
        pieces = []
        for st in range(0, idx.size, 100_000):
            pieces.append(np.abs(np.asarray(arr[idx[st:st+100_000]], dtype=np.float32).reshape(-1)))
        x = np.concatenate(pieces) if pieces else np.array([], dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    v = float(np.nanpercentile(x, percentile))
    if not np.isfinite(v) or v <= 1e-12:
        v = float(np.nanmax(x)) if x.size else 1.0
    if not np.isfinite(v) or v <= 1e-12:
        v = 1.0
    return v


def q_scale_pass(t_arr: np.ndarray, I_arr: np.ndarray, n_time: int, chunk: int) -> float:
    q = 0.0
    qmax = 0.0
    prev_t = float(t_arr[0])
    for st in range(0, n_time, chunk):
        ed = min(n_time, st + chunk)
        t = one_d_get(t_arr, st, ed, n_time, "t").astype(np.float64)
        I = one_d_get(I_arr, st, ed, n_time, "I").astype(np.float64)
        if t.size == 0:
            continue
        dt = np.diff(t, prepend=prev_t)
        # match original rectangle-rule spirit.
        q_series = q + np.cumsum(I * dt) / 3600.0
        if q_series.size:
            qmax = max(qmax, float(np.nanmax(np.abs(q_series))))
            q = float(q_series[-1])
            prev_t = float(t[-1])
    return qmax if np.isfinite(qmax) and qmax > 1e-12 else 1.0


def make_features_chunk(t: np.ndarray, I: np.ndarray, voltage: np.ndarray, temp: np.ndarray, qn: np.ndarray, t0: float, span: float, I_scale: float, vmean: float, vstd: float, tmean: float, tstd: float, profile_index: int, profile_count: int, include_onehot: bool) -> Tuple[np.ndarray, List[str]]:
    tn = ((t - t0) / span).astype(np.float32)
    In = (I / I_scale).astype(np.float32)
    # dI within chunk; first point uses 0 local derivative. This only affects one boundary point per chunk.
    dI = np.diff(I, prepend=I[0]).astype(np.float32) / I_scale
    vn = ((voltage - vmean) / vstd).astype(np.float32)
    Tn = ((temp - tmean) / tstd).astype(np.float32)
    charge = (I > max(1e-9, 0.001 * I_scale)).astype(np.float32)
    discharge = (I < -max(1e-9, 0.001 * I_scale)).astype(np.float32)
    rest = 1.0 - np.maximum(charge, discharge)
    base = [tn, tn**2, np.sin(2*np.pi*tn).astype(np.float32), np.cos(2*np.pi*tn).astype(np.float32), In, np.abs(In), dI.astype(np.float32), qn.astype(np.float32), vn, Tn, charge, rest, discharge]
    names = ['t_norm','t_norm2','sin_t','cos_t','I_norm','absI_norm','dI_norm','q_norm','voltage_exp_norm_local','temperature_norm_local','is_charge','is_rest','is_discharge']
    X = np.stack(base, axis=1).astype(np.float32)
    if include_onehot:
        oh = np.zeros((X.shape[0], profile_count), dtype=np.float32)
        if 0 <= profile_index < profile_count:
            oh[:, profile_index] = 1.0
        X = np.concatenate([X, oh], axis=1)
        names.extend([f'profile_onehot_{i:02d}' for i in range(profile_count)])
    return X, names


def standardize_x(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean[None, :]) / std[None, :]).astype(np.float32)


def unstandardize_y(Yn: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (Yn * std[None, :] + mean[None, :]).astype(np.float32)


def predict_chunk(model: Any, X: np.ndarray, state: Mapping[str, Any], device: Any, batch_size: int) -> np.ndarray:
    x_mean = np.asarray(state['x_mean'], dtype=np.float32)
    x_std = np.asarray(state['x_std'], dtype=np.float32)
    y_mean = np.asarray(state['y_mean'], dtype=np.float32)
    y_std = np.asarray(state['y_std'], dtype=np.float32)
    outs = []
    model.eval()
    with torch.no_grad():
        for st in range(0, X.shape[0], batch_size):
            xb = standardize_x(X[st:st+batch_size], x_mean, x_std)
            xt = torch.from_numpy(xb).to(device)
            yp = model(xt).detach().cpu().numpy()
            outs.append(unstandardize_y(yp, y_mean, y_std))
    return np.concatenate(outs, axis=0).astype(np.float32)


def apply_projection(y: np.ndarray, slices: Mapping[str, Tuple[int, int]], theta_min: float, theta_max: float) -> np.ndarray:
    out = y.copy()
    for key in ["theta_a", "theta_c"]:
        s, e = slices[key]
        out[:, s:e] = np.clip(out[:, s:e], theta_min, theta_max)
    return out


def npy_save_small(path: Path, arr: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        np.save(f, arr)


def make_npz_from_npy_members(out_npz: Path, members: Mapping[str, Path], compress: bool = True) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    tmp_npz = out_npz.with_suffix(out_npz.suffix + ".tmp")
    if tmp_npz.exists():
        tmp_npz.unlink()
    with zipfile.ZipFile(tmp_npz, "w", compression=mode, allowZip64=True, compresslevel=1 if compress else None) as zf:
        for key, p in members.items():
            zf.write(p, arcname=f"{key}.npy")
    if out_npz.exists():
        out_npz.unlink()
    tmp_npz.replace(out_npz)


def load_existing_prediction_metrics(pred_file: Path, slices: Mapping[str, Tuple[int,int]]) -> Dict[str, Any]:
    with np.load(pred_file, allow_pickle=True) as z:
        yt = np.asarray(z["y_true"], dtype=np.float32)
        yp = np.asarray(z["y_pred"], dtype=np.float32)
        t = np.asarray(z["t_global_s"], dtype=np.float32) if "t_global_s" in z else np.arange(yt.shape[0], dtype=np.float32)
        pid = str(np.asarray(z.get("profile_id", pred_file.stem)).reshape(-1)[0]) if "profile_id" in z else pred_file.stem
    ps = ProfileStats(slices)
    # Process in pieces to avoid spikes if file is large.
    for st in range(0, yt.shape[0], 200_000):
        ed = min(yt.shape[0], st+200_000)
        ps.update(yt[st:ed], yp[st:ed])
    row = ps.as_dict()
    row.update({"profile_id": pid, "prediction_file": str(pred_file), "n_eval": int(yt.shape[0]), "status": "EXISTING"})
    b, pr, cell = parse_batch_protocol(pid)
    row.update({"batch": b, "protocol": pr, "cell_key": cell})
    return row


def extract_required_arrays(npz_path: Path, tmp_dir: Path) -> Dict[str, np.ndarray]:
    mapping = {
        "t": TIME_KEYS,
        "I": I_KEYS,
        "voltage": VOLTAGE_KEYS,
        "temp": TEMP_KEYS,
        "theta_a": THETA_A_KEYS,
        "theta_c": THETA_C_KEYS,
        "phie": PHIE_KEYS,
        "phis_c": PHIS_KEYS,
    }
    out: Dict[str, np.ndarray] = {}
    for key, keys in mapping.items():
        required = key not in ("temp", "voltage")
        path = extract_member_to_npy(npz_path, keys, tmp_dir, required=required)
        if path is None:
            out[key] = None  # type: ignore
        else:
            out[key] = load_mmap_npy(path)
    return out


def build_y_true_chunk(arrs: Mapping[str, np.ndarray], start: int, end: int, n_time: int) -> np.ndarray:
    ta = orient_get(arrs["theta_a"], start, end, n_time, "theta_a")
    tc = orient_get(arrs["theta_c"], start, end, n_time, "theta_c")
    phie = one_d_get(arrs["phie"], start, end, n_time, "phie").reshape(-1, 1)
    phis = one_d_get(arrs["phis_c"], start, end, n_time, "phis_c").reshape(-1, 1)
    return np.concatenate([ta, tc, phie, phis], axis=1).astype(np.float32)


def infer_one_fullprofile(npz_path: Path, root: Path, pred_file: Path, model: Any, state: Mapping[str, Any], device: Any, profile_index: int, profile_count: int, primary_mode: str, batch_size: int, chunk_size: int, tmp_root: Path, compress_npz: bool = True) -> Dict[str, Any]:
    pid = profile_id_from_npz(npz_path, root)
    tmp_dir = tmp_root / safe_name(pid)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    t_start_wall = time.time()
    try:
        arrs = extract_required_arrays(npz_path, tmp_dir)
        t_arr = arrs["t"]
        n_time = int(np.asarray(t_arr).reshape(-1).shape[0])
        I_arr = arrs["I"]
        voltage_arr = arrs.get("voltage")
        temp_arr = arrs.get("temp")
        # Fill optional arrays with lightweight .npy if missing.
        if voltage_arr is None:
            # Use phis_c mean as fallback voltage.
            phis = arrs["phis_c"]
            # mean streaming
            vmean, _ = streaming_mean_std(phis, n_time, chunk_size, fill_if_bad=0.0)
            vpath = tmp_dir / "voltage_filled.npy"
            vm = np.lib.format.open_memmap(vpath, mode="w+", dtype=np.float32, shape=(n_time,))
            vm[:] = vmean
            del vm
            voltage_arr = np.load(vpath, mmap_mode="r")
        if temp_arr is None:
            tpath = tmp_dir / "temp_filled.npy"
            tm = np.lib.format.open_memmap(tpath, mode="w+", dtype=np.float32, shape=(n_time,))
            tm[:] = 25.0
            del tm
            temp_arr = np.load(tpath, mmap_mode="r")
        t0 = float(t_arr[0])
        span = float(t_arr[n_time-1] - t_arr[0]) if n_time > 1 else 1.0
        if not np.isfinite(span) or span <= 0: span = 1.0
        I_scale = streaming_abs_percentile(I_arr, n_time, chunk_size)
        vmean, vstd = streaming_mean_std(voltage_arr, n_time, chunk_size, fill_if_bad=0.0)
        tempmean, tempstd = streaming_mean_std(temp_arr, n_time, chunk_size, fill_if_bad=25.0)
        qscale = q_scale_pass(t_arr, I_arr, n_time, chunk_size)
        target_slices = normalize_slices(state["target_slices"])
        output_dim = int(state["output_dim"])
        y_true_path = tmp_dir / "y_true.npy"
        y_pred_path = tmp_dir / "y_pred.npy"
        t_out_path = tmp_dir / "t_global_s.npy"
        y_true_mm = np.lib.format.open_memmap(y_true_path, mode="w+", dtype=np.float32, shape=(n_time, output_dim))
        y_pred_mm = np.lib.format.open_memmap(y_pred_path, mode="w+", dtype=np.float32, shape=(n_time, output_dim))
        t_mm = np.lib.format.open_memmap(t_out_path, mode="w+", dtype=np.float32, shape=(n_time,))
        stats = ProfileStats(target_slices)
        prev_t = float(t_arr[0])
        q = 0.0
        feature_names_saved = None
        include_onehot = bool(state.get("include_profile_onehot", True))
        for st in range(0, n_time, chunk_size):
            ed = min(n_time, st + chunk_size)
            t = one_d_get(t_arr, st, ed, n_time, "t")
            I = one_d_get(I_arr, st, ed, n_time, "I")
            V = one_d_get(voltage_arr, st, ed, n_time, "voltage")
            T = one_d_get(temp_arr, st, ed, n_time, "temp")
            dt = np.diff(t.astype(np.float64), prepend=prev_t)
            q_series = q + np.cumsum(I.astype(np.float64) * dt) / 3600.0
            qn = (q_series / qscale).astype(np.float32)
            if q_series.size:
                q = float(q_series[-1])
                prev_t = float(t[-1])
            X, fnames = make_features_chunk(t, I, V, T, qn, t0, span, I_scale, vmean, vstd, tempmean, tempstd, profile_index, profile_count, include_onehot)
            if feature_names_saved is None:
                feature_names_saved = fnames
            y_true = build_y_true_chunk(arrs, st, ed, n_time)
            y_pred = predict_chunk(model, X, state, device, batch_size=batch_size)
            if primary_mode == "projected":
                y_pred = apply_projection(y_pred, target_slices, theta_min=1e-4, theta_max=1.0-1e-4)
            if y_true.shape[1] != output_dim or y_pred.shape[1] != output_dim:
                raise ValueError(f"output_dim mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}, expected={output_dim}")
            y_true_mm[st:ed, :] = y_true
            y_pred_mm[st:ed, :] = y_pred
            t_mm[st:ed] = t.astype(np.float32)
            stats.update(y_true, y_pred)
            print(f"[D16-P5A v6] {pid}: chunk {st}:{ed}/{n_time}", flush=True)
        # Flush memmaps
        del y_true_mm, y_pred_mm, t_mm
        # Small metadata npy files.
        target_names_path = tmp_dir / "target_names.npy"
        feature_names_path = tmp_dir / "feature_names.npy"
        profile_id_path = tmp_dir / "profile_id.npy"
        meta_path = tmp_dir / "d16_meta.npy"
        npy_save_small(target_names_path, np.array(state.get("target_names", []), dtype=object))
        npy_save_small(feature_names_path, np.array(feature_names_saved or state.get("feature_names", []), dtype=object))
        npy_save_small(profile_id_path, np.array(pid))
        npy_save_small(meta_path, np.array(json.dumps({"prediction_mode": primary_mode, "profile_index": profile_index, "profile_count": profile_count})))
        members = {
            "t_global_s": t_out_path,
            "y_true": y_true_path,
            "y_pred": y_pred_path,
            "target_names": target_names_path,
            "feature_names": feature_names_path,
            "profile_id": profile_id_path,
            "d16_meta_json": meta_path,
        }
        make_npz_from_npy_members(pred_file, members, compress=compress_npz)
        row = stats.as_dict()
        b, pr, cell = parse_batch_protocol(pid)
        row.update({
            "profile_id": pid,
            "npz_path": str(npz_path),
            "prediction_file": str(pred_file),
            "batch": b,
            "protocol": pr,
            "cell_key": cell,
            "n_eval": int(n_time),
            "status": "GENERATED",
            "prediction_mode": primary_mode,
            "runtime_s": float(time.time() - t_start_wall),
        })
        sidecar = pred_file.with_suffix(".metrics.json")
        write_json(row, sidecar)
        return row
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def aggregate_metric_rows(rows: Sequence[Mapping[str, Any]], group_key: Optional[str] = None) -> List[Dict[str, Any]]:
    # Weighted by count for MAE/RMSE/Bias using available *_count. Corr is mean of finite corr, not pooled.
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for r in rows:
        k = "ALL" if group_key is None else str(r.get(group_key, "unknown"))
        groups.setdefault(k, []).append(r)
    out_rows = []
    prefixes = ["phis_c", "phie", "theta_a", "theta_c", "theta_a_mean", "theta_c_mean", "grad_a_surface_center", "grad_c_surface_center"]
    for g, rs in groups.items():
        out: Dict[str, Any] = {"group": g, "profile_count": len(rs)}
        if group_key is not None:
            out[group_key] = g
        for p in prefixes:
            total_count = 0
            mae_num = 0.0
            rmse_num = 0.0
            bias_num = 0.0
            max_abs = 0.0
            corr_vals = []
            for r in rs:
                c = r.get(f"{p}_count")
                if c is None:
                    continue
                c = int(c)
                if c <= 0:
                    continue
                total_count += c
                mae = r.get(f"{p}_mae")
                rmse = r.get(f"{p}_rmse")
                bias = r.get(f"{p}_bias")
                mx = r.get(f"{p}_max_abs")
                corr = r.get(f"{p}_corr")
                if mae is not None: mae_num += float(mae) * c
                if rmse is not None: rmse_num += (float(rmse) ** 2) * c
                if bias is not None: bias_num += float(bias) * c
                if mx is not None: max_abs = max(max_abs, float(mx))
                if corr is not None and np.isfinite(float(corr)): corr_vals.append(float(corr))
            out[f"{p}_count"] = total_count
            out[f"{p}_mae"] = None if total_count <= 0 else float(mae_num / total_count)
            out[f"{p}_rmse"] = None if total_count <= 0 else float(math.sqrt(rmse_num / total_count))
            out[f"{p}_bias"] = None if total_count <= 0 else float(bias_num / total_count)
            out[f"{p}_max_abs"] = None if total_count <= 0 else float(max_abs)
            out[f"{p}_corr_mean"] = None if not corr_vals else float(np.mean(corr_vals))
        theta_counts = sum(int(r.get("theta_total_count", 0) or 0) for r in rs)
        outside_num = sum(float(r.get("pred_theta_outside_fraction", 0) or 0) * int(r.get("theta_total_count", 0) or 0) for r in rs)
        out["pred_theta_outside_fraction_weighted"] = None if theta_counts <= 0 else float(outside_num / theta_counts)
        out_rows.append(out)
    return out_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D16-P5A v6: fill missing ALL55 full-profile predictions using chunked inference.")
    p.add_argument("--softlabel-dir", default=r"E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL")
    p.add_argument("--run-dir", default=r"E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_V4_D15_existing_on_ALL55")
    p.add_argument("--model-dir", default=r"E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p2_rg_precision_benchmark")
    p.add_argument("--filename", default="solution_softlabels.npz")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32768)
    p.add_argument("--chunk-size", type=int, default=200000)
    p.add_argument("--limit-missing", type=int, default=None)
    p.add_argument("--allow-overwrite", action="store_true")
    p.add_argument("--recompute-existing-metrics", action="store_true")
    p.add_argument("--primary-mode", choices=["projected", "raw"], default="projected")
    p.add_argument("--no-compress", action="store_true", help="Use uncompressed npz members for faster write but larger files.")
    return p.parse_args()


def model_file(model_dir: Path) -> Path:
    if (model_dir / "model" / "best_with_state.pt").exists():
        return model_dir / "model" / "best_with_state.pt"
    if (model_dir / "best_with_state.pt").exists():
        return model_dir / "best_with_state.pt"
    raise FileNotFoundError(f"Could not find D15 best_with_state.pt under {model_dir}")


def main() -> int:
    args = parse_args()
    if torch is None:
        raise RuntimeError("PyTorch is required for inference")
    soft_root = Path(args.softlabel_dir)
    run_dir = Path(args.run_dir)
    eval_dir = run_dir / "eval_full_profiles"
    pred_dir = eval_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = run_dir / "_tmp_v6_fullprofile"
    tmp_root.mkdir(parents=True, exist_ok=True)
    mf = model_file(Path(args.model_dir))
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(mf, map_location=device, weights_only=False)
    state = ck["state"]
    slices = normalize_slices(state["target_slices"])
    model = build_model(int(state["input_dim"]), int(state["output_dim"]), state["model_config"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    profile_ids = list(state.get("profile_ids", []))
    profile_count = len(profile_ids)
    files = discover_npz(soft_root, filename=args.filename)
    if not files:
        raise FileNotFoundError(f"No {args.filename} found under {soft_root}")
    routing_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    to_generate: List[Tuple[Path, str, Path, int, str]] = []
    for npz_path in files:
        pid = profile_id_from_npz(npz_path, soft_root)
        pred_file = pred_dir / f"{safe_name(pid)}_prediction.npz"
        idx, reason = choose_profile_index(pid, profile_ids)
        b, pr, cell = parse_batch_protocol(pid)
        routing_rows.append({"profile_id": pid, "batch": b, "protocol": pr, "cell_key": cell, "routed_profile_index": idx, "routed_seen_profile_id": profile_ids[idx] if profile_ids else "", "routing_reason": reason, "prediction_file": str(pred_file), "exists_before": pred_file.exists()})
        if pred_file.exists() and not args.allow_overwrite:
            # Keep existing prediction and compute/read metrics.
            side = pred_file.with_suffix(".metrics.json")
            if side.exists() and not args.recompute_existing_metrics:
                with side.open("r", encoding="utf-8") as f:
                    metric_rows.append(json.load(f))
            else:
                try:
                    row = load_existing_prediction_metrics(pred_file, slices)
                    write_json(row, side)
                    metric_rows.append(row)
                except Exception as exc:
                    failures.append({"profile_id": pid, "stage": "existing_metrics", "error": repr(exc), "prediction_file": str(pred_file)})
            continue
        if pred_file.exists() and args.allow_overwrite:
            # For fill-missing use, do not overwrite existing by default. But AllowOverwrite is used for run dir safety.
            side = pred_file.with_suffix(".metrics.json")
            if side.exists() and not args.recompute_existing_metrics:
                with side.open("r", encoding="utf-8") as f:
                    metric_rows.append(json.load(f))
            else:
                try:
                    row = load_existing_prediction_metrics(pred_file, slices)
                    write_json(row, side)
                    metric_rows.append(row)
                except Exception as exc:
                    failures.append({"profile_id": pid, "stage": "existing_metrics", "error": repr(exc), "prediction_file": str(pred_file)})
            continue
        to_generate.append((npz_path, pid, pred_file, idx, reason))
    if args.limit_missing is not None:
        to_generate = to_generate[:max(0, int(args.limit_missing))]
    print(f"[D16-P5A v6] soft profiles discovered={len(files)} existing metrics={len(metric_rows)} missing_to_generate={len(to_generate)}", flush=True)
    for npz_path, pid, pred_file, idx, reason in to_generate:
        try:
            print(f"[D16-P5A v6] GENERATE {pid} -> {pred_file}", flush=True)
            row = infer_one_fullprofile(npz_path, soft_root, pred_file, model, state, device, idx, profile_count, args.primary_mode, args.batch_size, args.chunk_size, tmp_root, compress_npz=not args.no_compress)
            row["routing_reason"] = reason
            row["routed_profile_index"] = idx
            row["routed_seen_profile_id"] = profile_ids[idx] if profile_ids else ""
            metric_rows.append(row)
        except Exception as exc:
            print(f"[D16-P5A v6] FAIL {pid}: {exc!r}", flush=True)
            failures.append({"profile_id": pid, "stage": "generate", "error": repr(exc), "npz_path": str(npz_path), "prediction_file": str(pred_file)})
    write_csv(routing_rows, eval_dir / "D16_P5A_V6_ROUTING_TABLE.csv")
    write_csv(metric_rows, eval_dir / "D16_P5A_METRICS_BY_PROFILE.csv")
    write_json(metric_rows, eval_dir / "D16_P5A_METRICS_BY_PROFILE.json")
    batch_rows = aggregate_metric_rows(metric_rows, group_key="batch")
    protocol_rows = aggregate_metric_rows(metric_rows, group_key="protocol")
    global_rows = aggregate_metric_rows(metric_rows, group_key=None)
    write_csv(batch_rows, eval_dir / "D16_P5A_BATCH_METRICS.csv")
    write_csv(protocol_rows, eval_dir / "D16_P5A_PROTOCOL_METRICS.csv")
    write_json(failures, eval_dir / "D16_P5A_FAILURES.json")
    pred_count = len(list(pred_dir.glob("*.npz")))
    final = {
        "stage": "D16-P5A v6 full-profile fill-missing transfer evaluation",
        "softlabel_dir": str(soft_root),
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
        "prediction_dir": str(pred_dir),
        "model_file": str(mf),
        "profile_count_discovered": len(files),
        "prediction_file_count_primary": pred_count,
        "profile_count_with_metrics": len(metric_rows),
        "generated_this_run": len(to_generate) - sum(1 for f in failures if f.get("stage") == "generate"),
        "failure_count": len(failures),
        "operational_status": "PASS" if pred_count == len(files) and len(failures) == 0 else "REVIEW" if pred_count > 0 else "FAIL",
        "global_metrics_weighted": global_rows[0] if global_rows else {},
        "notes": [
            "This is full-profile chunked inference. It fills missing prediction files without rerunning already successful profiles.",
            "Only eval_full_profiles/predictions is written; raw/projected duplicate directories are intentionally not created.",
            "primary_mode=projected clips theta_a/theta_c to the D15-P3B physical range but leaves phie/phis_c unchanged.",
        ],
        "failures": failures,
    }
    write_json(final, run_dir / "D16_P5A_V6_FINAL_SCORECARD.json")
    print("[D16-P5A v6] operational_status:", final["operational_status"], flush=True)
    print("[D16-P5A v6] prediction_file_count_primary:", pred_count, flush=True)
    print("[D16-P5A v6] wrote:", run_dir / "D16_P5A_V6_FINAL_SCORECARD.json", flush=True)
    return 0 if final["operational_status"] in {"PASS", "REVIEW"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
