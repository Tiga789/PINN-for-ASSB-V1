from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


OBS_TIME_KEYS = ["t_global_s", "time_s", "t_s", "t", "time"]
OBS_I_KEYS = ["I_profile", "current_A", "I_A", "current", "I"]
OBS_V_KEYS = ["voltage_exp", "voltage_V", "V_exp", "V", "voltage"]
OBS_T_KEYS = ["temperature_C", "temp_C", "T_C", "temperature_K", "T", "temperature"]
STATE_KEYS = ["theta_a", "theta_c", "cs_a", "cs_c", "phie", "phis_c"]
CORE_TARGETS = ["cs_a", "cs_c", "phie", "phis_c"]


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------


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


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv_rows(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
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
        rows = [{"empty": ""}]
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))


def torch_load_safe(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location=map_location)


def device_from_arg(arg: str) -> torch.device:
    if str(arg).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(arg))


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


# -----------------------------------------------------------------------------
# Model definition compatible with D17-G1.4/G2.1 checkpoint
# -----------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PhieConventionHead(nn.Module):
    def __init__(self, fused_dim: int, local_dim: int, profile_dim: int, out_dim: int = 1, width: int = 320, dropout: float = 0.04):
        super().__init__()
        self.local_dim = int(local_dim)
        self.profile_dim = int(profile_dim)
        self.out_dim = int(out_dim)
        obs_dim = self.local_dim + max(0, self.profile_dim)
        self.obs_basis = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, max(64, int(width) // 2)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(64, int(width) // 2), out_dim),
        )
        self.residual = nn.Sequential(
            nn.Linear(int(fused_dim) + obs_dim, int(width)),
            nn.SiLU(),
            ResidualBlock(int(width), dropout=float(dropout)),
            nn.LayerNorm(int(width)),
            nn.Linear(int(width), out_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(obs_dim, max(32, int(width) // 4)),
            nn.SiLU(),
            nn.Linear(max(32, int(width) // 4), out_dim),
            nn.Sigmoid(),
        )

    def forward(self, fused: torch.Tensor, local: torch.Tensor, profile: torch.Tensor | None) -> torch.Tensor:
        obs = local if profile is None else torch.cat([local, profile], dim=-1)
        return self.obs_basis(obs) + self.gate(obs) * self.residual(torch.cat([fused, obs], dim=-1))


class ValidationRobustObservedProfileSurrogate(nn.Module):
    def __init__(
        self,
        local_input_dim: int,
        profile_input_dim: int,
        target_slices: Mapping[str, Tuple[int, int]],
        width: int = 896,
        depth: int = 8,
        profile_width: int = 256,
        dropout: float = 0.05,
        phie_direct_width: int = 320,
    ):
        super().__init__()
        self.local_input_dim = int(local_input_dim)
        self.profile_input_dim = int(profile_input_dim)
        self.target_slices = OrderedDict((str(k), (int(v[0]), int(v[1]))) for k, v in target_slices.items())
        self.target_order = list(self.target_slices.keys())
        self.output_dim = max(b for _, b in self.target_slices.values())
        w = int(width)
        pw = int(profile_width)
        self.local_encoder = nn.Sequential(
            nn.Linear(self.local_input_dim, w),
            nn.SiLU(),
            ResidualBlock(w, dropout=float(dropout)),
            ResidualBlock(w, dropout=float(dropout)),
            nn.LayerNorm(w),
        )
        if self.profile_input_dim > 0:
            self.profile_encoder = nn.Sequential(
                nn.Linear(self.profile_input_dim, pw),
                nn.SiLU(),
                ResidualBlock(pw, dropout=float(dropout)),
                ResidualBlock(pw, dropout=float(dropout)),
                nn.LayerNorm(pw),
            )
            fused_in = w + pw
        else:
            self.profile_encoder = None
            fused_in = w
        layers: List[nn.Module] = [nn.Linear(fused_in, w), nn.SiLU()]
        for _ in range(max(0, int(depth))):
            layers.append(ResidualBlock(w, dropout=float(dropout)))
        layers.append(nn.LayerNorm(w))
        self.fusion = nn.Sequential(*layers)
        self.heads = nn.ModuleDict()
        for key, (a, b) in self.target_slices.items():
            if key == "phie":
                continue
            out_dim = int(b - a)
            half = max(96, w // 2)
            self.heads[key] = nn.Sequential(
                nn.Linear(w, half),
                nn.SiLU(),
                ResidualBlock(half, dropout=float(dropout)),
                nn.LayerNorm(half),
                nn.Linear(half, out_dim),
            )
        phie_a, phie_b = self.target_slices.get("phie", (0, 1))
        self.phie_head = PhieConventionHead(w, self.local_input_dim, self.profile_input_dim, max(1, int(phie_b - phie_a)), int(phie_direct_width), float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = x[:, : self.local_input_dim]
        lz = self.local_encoder(local)
        if self.profile_input_dim > 0:
            profile = x[:, self.local_input_dim : self.local_input_dim + self.profile_input_dim]
            pz = self.profile_encoder(profile) if self.profile_encoder is not None else None
            z = self.fusion(torch.cat([lz, pz], dim=-1))
        else:
            profile = None
            z = self.fusion(lz)
        chunks: List[torch.Tensor] = []
        for key in self.target_order:
            if key == "phie":
                chunks.append(self.phie_head(z, local, profile))
            else:
                chunks.append(self.heads[key](z))
        return torch.cat(chunks, dim=-1)


# -----------------------------------------------------------------------------
# Feature and data helpers compatible with G1/G2 training
# -----------------------------------------------------------------------------


def _first_existing_key(keys: Sequence[str], zfiles: Sequence[str]) -> Optional[str]:
    s = set(zfiles)
    for k in keys:
        if k in s:
            return k
    return None


def _to_1d(x: Any, dtype: Any = np.float32) -> np.ndarray:
    return np.asarray(x).astype(dtype).reshape(-1)


def _to_1d_any(x: Any) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def _as_time_radial(x: np.ndarray, n_time: int, key: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 1:
        if arr.shape[0] != n_time:
            raise ValueError(f"{key}: length {arr.shape[0]} != n_time {n_time}")
        return arr.reshape(n_time, 1).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{key}: expected 1D/2D, got {arr.shape}")
    if arr.shape[0] == n_time:
        return arr.astype(np.float32)
    if arr.shape[1] == n_time:
        return arr.T.astype(np.float32)
    raise ValueError(f"{key}: cannot orient {arr.shape} for n_time {n_time}")


def parse_cycle_spec(spec: str) -> Optional[set[int]]:
    s = str(spec).strip().lower()
    if s in {"all", "*", "全部"}:
        return None
    out: set[int] = set()
    for part in s.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            ia, ib = int(a), int(b)
            if ib < ia:
                ia, ib = ib, ia
            out.update(range(ia, ib + 1))
        else:
            out.add(int(part))
    return out


def find_record(records: Sequence[Mapping[str, Any]], batch: str, battery: str) -> Dict[str, Any]:
    bnorm = str(batch)
    if not bnorm.lower().startswith("batch-"):
        bnorm = f"Batch-{bnorm}"
    else:
        bnorm = "Batch-" + bnorm.split("-", 1)[1]
    batt = str(battery)
    if not batt.lower().startswith("battery-"):
        batt = f"battery-{batt}"
    target = f"{bnorm}_{batt}"
    for r in records:
        vals = [str(r.get(k, "")) for k in ["cell_uid", "canonical_cell_uid", "softlabel_dir", "softlabel_npz", "replay_npz"]]
        if any(target in v for v in vals):
            return dict(r)
        if str(r.get("batch")) == bnorm and str(r.get("battery")) == batt:
            return dict(r)
    raise ValueError(f"Could not find record for {target} in split manifest")


def load_split_records(split_manifest: str | Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    data = read_json(split_manifest, default={}) or {}
    return [dict(x) for x in data.get("records", [])], dict(data)


def load_semantics_map(path: str | Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv_rows(path)
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        for k in ["canonical_cell_uid", "cell_uid", "softlabel_npz"]:
            v = str(row.get(k, "")).strip()
            if v:
                out[v] = row
                try:
                    out[str(Path(v).resolve())] = row
                    out[Path(v).parent.name] = row
                except Exception:
                    pass
    return out


def semantics_for_record(record: Mapping[str, Any], sem_map: Mapping[str, Dict[str, str]]) -> Dict[str, str]:
    for key in [str(record.get("canonical_cell_uid", "")), str(record.get("cell_uid", "")), str(record.get("softlabel_npz", ""))]:
        if key in sem_map:
            return sem_map[key]
        try:
            k2 = str(Path(key).resolve())
            if k2 in sem_map:
                return sem_map[k2]
        except Exception:
            pass
    return {}


def parse_vocabs_from_checkpoint_feature_names(feature_names: Sequence[str], local_input_dim: int) -> Tuple[List[str], List[str]]:
    base_dim = max(0, int(local_input_dim) - 7)
    base = list(feature_names[:base_dim])
    protocols = [n[len("protocol_"):] for n in base if n.startswith("protocol_")]
    branches = [n[len("branch_"):] for n in base if n.startswith("branch_")]
    return protocols, branches


def _step_features(step_type: Optional[np.ndarray], I: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    n = I.size
    charge = np.zeros(n, dtype=np.float32)
    rest = np.zeros(n, dtype=np.float32)
    discharge = np.zeros(n, dtype=np.float32)
    if step_type is not None:
        try:
            st = np.asarray(step_type).reshape(-1)
            if st.size == n:
                for i, val in enumerate(st):
                    s = str(val).lower()
                    if "rest" in s or "静" in s or "搁" in s:
                        rest[i] = 1.0
                    elif "dis" in s or "放" in s:
                        discharge[i] = 1.0
                    elif "cha" in s or "充" in s:
                        charge[i] = 1.0
        except Exception:
            pass
    eps = max(1e-8, 0.001 * float(np.nanmax(np.abs(I)) + 1e-12)) if I.size else 1e-8
    unknown = (charge + rest + discharge) == 0
    charge[unknown & (I > eps)] = 1.0
    discharge[unknown & (I < -eps)] = 1.0
    rest[unknown & (np.abs(I) <= eps)] = 1.0
    return np.stack([charge, rest, discharge], axis=1).astype(np.float32), ["is_charge", "is_rest", "is_discharge"]


def _onehot(value: str, vocab: Sequence[str], prefix: str, n: int) -> Tuple[np.ndarray, List[str]]:
    names = [f"{prefix}_{v}" for v in vocab]
    out = np.zeros((n, len(vocab)), dtype=np.float32)
    if value in vocab:
        out[:, vocab.index(value)] = 1.0
    return out, names


def _cum_charge_ah(t: np.ndarray, I: np.ndarray) -> np.ndarray:
    if t.size == 0:
        return np.zeros_like(I, dtype=np.float32)
    dt = np.diff(t, prepend=t[0]).astype(np.float32)
    return (np.cumsum(I.astype(np.float32) * dt) / 3600.0).astype(np.float32)


def _cum_charge_norm(t: np.ndarray, I: np.ndarray) -> np.ndarray:
    q = _cum_charge_ah(t, I)
    scale = float(np.nanmax(np.abs(q))) if q.size else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(q, dtype=np.float32)
    return (q / scale).astype(np.float32)


def _zscore_local(x: np.ndarray) -> np.ndarray:
    m = float(np.nanmean(x)) if x.size else 0.0
    s = float(np.nanstd(x)) if x.size else 1.0
    if not np.isfinite(s) or s <= 1e-8:
        s = 1.0
    return ((x - m) / s).astype(np.float32)


def _build_base_features(t: np.ndarray, I: np.ndarray, V: np.ndarray, T: np.ndarray, step_type: Optional[np.ndarray], protocol: str, branch: str, protocol_vocab: Sequence[str], branch_vocab: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    n = int(t.size)
    span = float(t[-1] - t[0]) if n > 1 else 1.0
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    tn = ((t - t[0]) / span).astype(np.float32)
    I_scale = float(np.nanpercentile(np.abs(I), 99.5)) if I.size else 1.0
    if not np.isfinite(I_scale) or I_scale <= 1e-12:
        I_scale = 1.0
    In = (I / I_scale).astype(np.float32)
    dI = np.diff(In, prepend=In[0]).astype(np.float32)
    Vn = _zscore_local(V)
    dV = np.diff(Vn, prepend=Vn[0]).astype(np.float32)
    Tn = _zscore_local(T)
    qn = _cum_charge_norm(t, I)
    base = np.stack([tn, tn * tn, np.sqrt(np.clip(tn, 0.0, 1.0)).astype(np.float32), np.sin(2 * np.pi * tn).astype(np.float32), np.cos(2 * np.pi * tn).astype(np.float32), In, np.abs(In).astype(np.float32), dI, qn, Vn, dV, Tn], axis=1).astype(np.float32)
    names = ["t_norm", "t_norm2", "sqrt_t_norm", "sin_t", "cos_t", "I_norm", "absI_norm", "dI_norm", "q_norm", "voltage_exp_norm_local", "dV_norm_local", "temperature_norm_local"]
    step_feat, step_names = _step_features(step_type, I)
    proto, proto_names = _onehot(protocol, protocol_vocab, "protocol", n)
    br, br_names = _onehot(branch, branch_vocab, "branch", n)
    return np.concatenate([base, step_feat, proto, br], axis=1).astype(np.float32), names + step_names + proto_names + br_names


def _safe_stat(x: np.ndarray, fn: str, default: float = 0.0) -> float:
    try:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float(default)
        if fn == "mean": return float(np.mean(arr))
        if fn == "std": return float(np.std(arr))
        if fn == "min": return float(np.min(arr))
        if fn == "max": return float(np.max(arr))
        if fn.startswith("q"):
            return float(np.quantile(arr, float(fn[1:]) / 100.0))
    except Exception:
        return float(default)
    return float(default)


def _profile_summary_features(t: np.ndarray, I: np.ndarray, V: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, List[str]]:
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


def _local_observed_features(t: np.ndarray, I: np.ndarray, V: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    q = _cum_charge_ah(t, I)
    dV = np.diff(V, prepend=V[0]).astype(np.float32) if V.size else np.zeros_like(I, dtype=np.float32)
    dI = np.diff(I, prepend=I[0]).astype(np.float32) if I.size else np.zeros_like(I, dtype=np.float32)
    X = np.stack([I, np.abs(I), dI, V, dV, T, q], axis=1).astype(np.float32)
    names = ["I_A_abs", "absI_A_abs", "dI_A_abs", "voltage_exp_V_abs", "dV_exp_V_abs", "temperature_C_abs", "q_Ah_abs"]
    return X, names


def build_augmented_features(t: np.ndarray, I: np.ndarray, V: np.ndarray, T: np.ndarray, step_type: Optional[np.ndarray], protocol: str, branch: str, protocol_vocab: Sequence[str], branch_vocab: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    base, base_names = _build_base_features(t, I, V, T, step_type, protocol, branch, protocol_vocab, branch_vocab)
    local, local_names = _local_observed_features(t, I, V, T)
    pfeat, pnames = _profile_summary_features(t, I, V, T)
    repeated = np.repeat(pfeat.reshape(1, -1), t.size, axis=0).astype(np.float32)
    return np.concatenate([base, local, repeated], axis=1).astype(np.float32), base_names + local_names + pnames


def load_candidate_checkpoint(candidate_dir: str | Path, candidate_summary: str | Path | None = None, checkpoint: str | Path | None = None) -> Tuple[Dict[str, Any], Path, Dict[str, Any]]:
    summary: Dict[str, Any] = {}
    if candidate_summary and Path(candidate_summary).exists():
        summary = read_json(candidate_summary, default={}) or {}
    elif candidate_dir:
        for name in ["D17_G21_P4D_BRANCH_REPAIR_SUMMARY.json", "D17_G2_HELDOUT_SURROGATE_EXPANSION_SUMMARY.json", "D17_G61_FULL_CYCLE_COVERAGE_REPAIR_SUMMARY.json", "D17_G61_CANDIDATE_FOR_G6_SUMMARY.json"]:
            p = Path(candidate_dir) / name
            if p.exists():
                summary = read_json(p, default={}) or {}
                break
    if checkpoint:
        ckpt_path = Path(checkpoint)
    else:
        files = summary.get("files", {}) if isinstance(summary.get("files"), Mapping) else {}
        ckpt_path = Path(str(files.get("best_model_pt") or ""))
        if not ckpt_path.exists():
            ckpt_path = Path(candidate_dir) / "model" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Cannot find checkpoint. Tried: {ckpt_path}")
    ckpt = torch_load_safe(ckpt_path, map_location="cpu")
    return ckpt, ckpt_path, summary


def build_model_from_checkpoint(ckpt: Mapping[str, Any], device: torch.device) -> nn.Module:
    cfg = dict(ckpt.get("config") or {})
    model_cfg = dict(cfg.get("model") or {})
    target_slices = {str(k): (int(v[0]), int(v[1])) for k, v in dict(ckpt.get("target_slices") or {}).items()}
    model = ValidationRobustObservedProfileSurrogate(
        local_input_dim=int(ckpt.get("local_input_dim")),
        profile_input_dim=int(ckpt.get("profile_input_dim", 0)),
        target_slices=target_slices,
        width=int(model_cfg.get("width", 960)),
        depth=int(model_cfg.get("depth", 8)),
        profile_width=int(model_cfg.get("profile_width", 288)),
        dropout=float(model_cfg.get("dropout", 0.05)),
        phie_direct_width=int(model_cfg.get("phie_direct_width", 384)),
    ).to(device)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if state is None:
        raise KeyError("checkpoint has no model_state_dict/state_dict")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_array(model: nn.Module, X: np.ndarray, ckpt: Mapping[str, Any], device: torch.device, batch_size: int = 8192) -> np.ndarray:
    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std = np.asarray(ckpt["y_std"], dtype=np.float32)
    x_std = x_std.copy()
    x_std[~np.isfinite(x_std) | (np.abs(x_std) < 1e-8)] = 1.0
    Xn = ((X - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)).astype(np.float32)
    Xn[~np.isfinite(Xn)] = 0.0
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, Xn.shape[0], int(batch_size)):
            xb = torch.as_tensor(Xn[i : i + int(batch_size)], dtype=torch.float32, device=device)
            outs.append(model(xb).detach().cpu().numpy())
    yn = np.concatenate(outs, axis=0) if outs else np.zeros((0, y_mean.size), dtype=np.float32)
    return (yn * y_std.reshape(1, -1) + y_mean.reshape(1, -1)).astype(np.float32)


def _find_key(z: np.lib.npyio.NpzFile, keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in z.files:
            return k
    return None


def _interp_obs_from_replay(replay_npz: str | Path, target_t: np.ndarray, key_candidates: Sequence[str], fill: float) -> Tuple[str, np.ndarray]:
    p = Path(replay_npz)
    if not p.exists():
        return "missing_replay_filled", np.full(target_t.size, fill, dtype=np.float32)
    with np.load(p, allow_pickle=True) as z:
        tk = _find_key(z, OBS_TIME_KEYS)
        kk = _find_key(z, key_candidates)
        if kk is None:
            return "missing_key_filled", np.full(target_t.size, fill, dtype=np.float32)
        y = _to_1d(z[kk])
        if tk is not None:
            rt = _to_1d(z[tk])
        else:
            rt = None
    if y.size == target_t.size:
        return f"replay:{kk}:same_length", y.astype(np.float32)
    if rt is None or rt.size != y.size:
        x_old = np.linspace(0, 1, y.size, dtype=np.float32)
        x_new = np.linspace(0, 1, target_t.size, dtype=np.float32)
        return f"replay:{kk}:index_interp", np.interp(x_new, x_old, y).astype(np.float32)
    order = np.argsort(rt)
    x = rt[order]
    yy = y[order]
    good = np.isfinite(x) & np.isfinite(yy)
    x = x[good]
    yy = yy[good]
    if x.size == 0:
        return f"replay:{kk}:filled_no_finite", np.full(target_t.size, fill, dtype=np.float32)
    ux, idx = np.unique(x, return_index=True)
    yy = yy[idx]
    if ux.size == 1:
        return f"replay:{kk}:constant_time", np.full(target_t.size, float(yy[0]), dtype=np.float32)
    return f"replay:{kk}:time_interp", np.interp(target_t, ux, yy, left=float(yy[0]), right=float(yy[-1])).astype(np.float32)


def _linear_downsample_indices(n: int, max_points: int) -> np.ndarray:
    if max_points and max_points > 0 and n > max_points:
        return np.linspace(0, n - 1, int(max_points)).round().astype(np.int64)
    return np.arange(n, dtype=np.int64)


def load_selected_cycle_data(
    record: Mapping[str, Any],
    sem_row: Mapping[str, str],
    cycles: str,
    protocol_vocab: Sequence[str],
    branch_vocab: Sequence[str],
    metric_targets: Sequence[str],
    max_points_per_cycle: int = 0,
    prefer_replay_observed: bool = True,
) -> Dict[str, Any]:
    soft_path = Path(str(record.get("softlabel_npz") or sem_row.get("softlabel_npz") or ""))
    if not soft_path.exists():
        raise FileNotFoundError(f"Missing softlabel_npz: {soft_path}")
    replay_path = Path(str(record.get("replay_npz") or ""))
    need_keys = set(OBS_TIME_KEYS + OBS_I_KEYS + OBS_V_KEYS + OBS_T_KEYS + ["cycle_id", "step_type", "r_a", "r_c", "r_grid_a", "r_grid_c", "r", "r_grid"])
    need_keys.update(metric_targets)
    need_keys.update(["theta_a", "theta_c", "cs_a", "cs_c", "phie", "phis_c"])  # cheap to include names; arrays load lazily by z[k]
    with np.load(soft_path, allow_pickle=True) as z:
        tk = _find_key(z, OBS_TIME_KEYS)
        if tk is None:
            raise KeyError(f"No time key found in {soft_path}")
        t_full = _to_1d(z[tk])
        ck = _find_key(z, ["cycle_id", "cycle", "cycle_index"])
        if ck is None:
            raise KeyError(f"No cycle_id key found in {soft_path}")
        cyc_raw = _to_1d_any(z[ck])
        try:
            cyc_full = cyc_raw.astype(np.int64)
        except Exception:
            cyc_full = np.asarray([int(float(str(x))) for x in cyc_raw], dtype=np.int64)
        requested = parse_cycle_spec(cycles)
        if requested is None:
            mask = np.ones(cyc_full.shape[0], dtype=bool)
        else:
            mask = np.isin(cyc_full, np.asarray(sorted(requested), dtype=np.int64))
        idx = np.where(mask)[0].astype(np.int64)
        if idx.size == 0:
            avail = np.unique(cyc_full)
            raise ValueError(f"Cycle selection {cycles!r} returned zero time points. Available cycle range: {int(avail[0])}-{int(avail[-1])}, count={avail.size}")
        if max_points_per_cycle and max_points_per_cycle > 0:
            keep: List[np.ndarray] = []
            for c in sorted(np.unique(cyc_full[idx])):
                sub = idx[cyc_full[idx] == c]
                keep.append(sub[_linear_downsample_indices(sub.size, int(max_points_per_cycle))])
            idx = np.concatenate(keep).astype(np.int64)
            idx.sort()
        t = t_full[idx].astype(np.float32)
        cyc = cyc_full[idx].astype(np.int64)
        step_type = None
        if "step_type" in z.files:
            st = np.asarray(z["step_type"]).reshape(-1)
            if st.size == t_full.size:
                step_type = st[idx]
        # Observed fields: prefer replay interpolation, otherwise use observed copies inside softlabel NPZ.
        def get_soft_obs(cands: Sequence[str], fill: float) -> Tuple[str, np.ndarray]:
            kk = _find_key(z, cands)
            if kk is None:
                return "soft_missing_filled", np.full(t.size, fill, dtype=np.float32)
            arr = _to_1d(z[kk])
            if arr.size == t_full.size:
                return f"soft:{kk}:indexed", arr[idx].astype(np.float32)
            if arr.size == t.size:
                return f"soft:{kk}:same_selected", arr.astype(np.float32)
            return f"soft:{kk}:size_mismatch_filled", np.full(t.size, fill, dtype=np.float32)
        if prefer_replay_observed and replay_path.exists():
            I_src, I = _interp_obs_from_replay(replay_path, t, OBS_I_KEYS, 0.0)
            V_src, V = _interp_obs_from_replay(replay_path, t, OBS_V_KEYS, 0.0)
            T_src, T = _interp_obs_from_replay(replay_path, t, OBS_T_KEYS, 25.0)
        else:
            I_src, I = get_soft_obs(OBS_I_KEYS, 0.0)
            V_src, V = get_soft_obs(OBS_V_KEYS, 0.0)
            T_src, T = get_soft_obs(OBS_T_KEYS, 25.0)
        protocol = str(record.get("protocol") or sem_row.get("protocol") or "UNKNOWN")
        branch = str(sem_row.get("semantic_branch") or "UNKNOWN_OR_MIXED_BRANCH")
        X, feature_names = build_augmented_features(t, I, V, T, step_type, protocol, branch, protocol_vocab, branch_vocab)
        targets: Dict[str, np.ndarray] = {}
        radial: Dict[str, np.ndarray] = {}
        # Target arrays are loaded one-by-one and immediately sliced.
        for key in sorted(set(metric_targets)):
            if key not in z.files:
                continue
            arr = _as_time_radial(z[key], t_full.size, key)[idx]
            targets[key] = arr.astype(np.float32)
        for side, n_key in [("a", "cs_a"), ("c", "cs_c")]:
            nr = int(targets[n_key].shape[1]) if n_key in targets else 17
            rkey = None
            for cand in [f"r_{side}", f"r_grid_{side}", f"r_{side}_norm", f"r_grid_norm_{side}", "r", "r_grid"]:
                if cand in z.files:
                    rr = np.asarray(z[cand], dtype=np.float32).reshape(-1)
                    if rr.size == nr:
                        rkey = cand
                        radial[side] = rr.astype(np.float32)
                        break
            if side not in radial:
                radial[side] = np.linspace(0.0, 1.0, nr, dtype=np.float32)
                rkey = "synthetic_linspace_0_1"
            radial[f"{side}_source"] = np.array(str(rkey))
    return {
        "softlabel_npz": str(soft_path),
        "replay_npz": str(replay_path),
        "t": t,
        "cycle_id": cyc,
        "I": I.astype(np.float32),
        "V": V.astype(np.float32),
        "T": T.astype(np.float32),
        "step_type": step_type,
        "X": X,
        "feature_names": feature_names,
        "targets": targets,
        "radial": radial,
        "protocol": protocol,
        "branch": branch,
        "observed_sources": {"I": I_src, "V": V_src, "T": T_src},
        "selected_cycles": sorted(set(int(x) for x in cyc.tolist())),
    }


def slice_prediction_by_targets(pred: np.ndarray, target_slices: Mapping[str, Tuple[int, int]], targets: Sequence[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key in targets:
        if key not in target_slices:
            continue
        a, b = target_slices[key]
        out[key] = pred[:, a:b].astype(np.float32)
    return out


def r2_score(y: np.ndarray, p: np.ndarray) -> float:
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    pv = np.asarray(p, dtype=np.float64).reshape(-1)
    mask = np.isfinite(yv) & np.isfinite(pv)
    yv = yv[mask]
    pv = pv[mask]
    if yv.size == 0:
        return float("nan")
    sse = float(np.sum((pv - yv) ** 2))
    sst = float(np.sum((yv - float(np.mean(yv))) ** 2))
    if sst <= 1e-20:
        return 1.0 if sse <= 1e-20 else float("nan")
    return 1.0 - sse / sst


def metrics_for_target(y: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    pv = np.asarray(p, dtype=np.float64).reshape(-1)
    mask = np.isfinite(yv) & np.isfinite(pv)
    yv = yv[mask]
    pv = pv[mask]
    if yv.size == 0:
        return {"n_points": 0, "r2": float("nan")}
    err = pv - yv
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    tr = float(np.max(yv) - np.min(yv))
    std = float(np.std(yv))
    return {
        "n_points": int(yv.size),
        "r2": r2_score(yv, pv),
        "mae": mae,
        "rmse": rmse,
        "nmae": mae / max(abs(tr), 1e-12),
        "nrmse": rmse / max(abs(tr), 1e-12),
        "bias": float(np.mean(err)),
        "target_min": float(np.min(yv)),
        "target_max": float(np.max(yv)),
        "target_range": tr,
        "target_std": std,
        "pred_min": float(np.min(pv)),
        "pred_max": float(np.max(pv)),
        "pred_std": float(np.std(pv)),
    }


def make_metrics_rows(data: Mapping[str, Any], pred_by_target: Mapping[str, np.ndarray], record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    true_targets: Dict[str, np.ndarray] = data["targets"]
    cycles = np.asarray(data["cycle_id"]).reshape(-1)
    for key, y in true_targets.items():
        if key not in pred_by_target:
            continue
        p = pred_by_target[key]
        global_m = metrics_for_target(y, p)
        rows.append({
            "level": "selected_range_global",
            "split": record.get("split", ""),
            "canonical_cell_uid": record.get("canonical_cell_uid", ""),
            "cell_uid": record.get("cell_uid", ""),
            "protocol": data.get("protocol"),
            "semantic_branch": data.get("branch"),
            "cycles": ",".join(str(c) for c in data.get("selected_cycles", [])),
            "target": key,
            **global_m,
        })
        for c in sorted(set(int(x) for x in cycles.tolist())):
            idx = cycles == c
            cm = metrics_for_target(y[idx], p[idx])
            rows.append({
                "level": "cycle",
                "split": record.get("split", ""),
                "canonical_cell_uid": record.get("canonical_cell_uid", ""),
                "cell_uid": record.get("cell_uid", ""),
                "protocol": data.get("protocol"),
                "semantic_branch": data.get("branch"),
                "cycle_id": int(c),
                "target": key,
                **cm,
            })
    return rows


def aggregate_selected_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    global_rows = [r for r in rows if r.get("level") == "selected_range_global"]
    vals = [safe_float(r.get("r2")) for r in global_rows]
    vals = [v for v in vals if math.isfinite(v)]
    out: Dict[str, Any] = {
        "selected_target_r2_mean": float(np.mean(vals)) if vals else float("nan"),
        "selected_target_r2_min": float(np.min(vals)) if vals else float("nan"),
    }
    for r in global_rows:
        key = str(r.get("target"))
        out[f"{key}_r2"] = safe_float(r.get("r2"))
        out[f"{key}_nmae"] = safe_float(r.get("nmae"))
        out[f"{key}_nrmse"] = safe_float(r.get("nrmse"))
    return out


def save_temp_prediction_npz(path: str | Path, data: Mapping[str, Any], pred_by_target: Mapping[str, np.ndarray], keep_true: bool = True) -> None:
    arrays: Dict[str, Any] = {
        "t_global_s": np.asarray(data["t"], dtype=np.float32),
        "cycle_id": np.asarray(data["cycle_id"], dtype=np.int64),
        "I_profile": np.asarray(data["I"], dtype=np.float32),
        "voltage_exp": np.asarray(data["V"], dtype=np.float32),
        "temperature_C": np.asarray(data["T"], dtype=np.float32),
        "protocol": np.array(str(data.get("protocol", ""))),
        "semantic_branch": np.array(str(data.get("branch", ""))),
    }
    for key, arr in pred_by_target.items():
        arrays[f"{key}_pred"] = arr.astype(np.float32)
        if keep_true and key in data["targets"]:
            arrays[f"{key}_true_report_only"] = np.asarray(data["targets"][key], dtype=np.float32)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)


def sample_for_plot(t: np.ndarray, pred: np.ndarray, true: np.ndarray, max_points: int) -> np.ndarray:
    n = int(t.size)
    return _linear_downsample_indices(n, max_points)


def plot_3d_pair(
    target: str,
    t: np.ndarray,
    r: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    metrics: Mapping[str, Any],
    title_prefix: str,
    save_path: str | Path | None = None,
    show: bool = True,
    time_axis: str = "relative",
    max_time_points: int = 1200,
    pred_cmap: str = "coolwarm",
    true_cmap: str = "viridis",
) -> None:
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    matplotlib.rcParams["font.family"] = "Times New Roman"
    idx = sample_for_plot(t, pred, true, max_time_points)
    tt = np.asarray(t, dtype=np.float64)[idx]
    if time_axis == "relative":
        tt = tt - tt[0]
    rr = np.asarray(r, dtype=np.float64).reshape(-1)
    P = np.asarray(pred, dtype=np.float64)[idx]
    Y = np.asarray(true, dtype=np.float64)[idx]
    Tgrid, Rgrid = np.meshgrid(tt, rr, indexing="ij")
    zmin = float(np.nanmin([np.nanmin(P), np.nanmin(Y)]))
    zmax = float(np.nanmax([np.nanmax(P), np.nanmax(Y)]))

    fig = plt.figure(figsize=(14, 6), dpi=120)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    s1 = ax1.plot_surface(Tgrid, Rgrid, P, cmap=pred_cmap, linewidth=0, antialiased=True, rstride=1, cstride=1)
    s2 = ax2.plot_surface(Tgrid, Rgrid, Y, cmap=true_cmap, linewidth=0, antialiased=True, rstride=1, cstride=1)
    for ax, name in [(ax1, "PINN prediction"), (ax2, "Soft-label truth")]:
        ax.set_title(name, fontname="Times New Roman")
        ax.set_xlabel("Time (s)", fontname="Times New Roman")
        ax.set_ylabel("Radial coordinate r/R", fontname="Times New Roman")
        ax.set_zlabel(f"{target} concentration" if target.startswith("cs") else target, fontname="Times New Roman")
        ax.set_zlim(zmin, zmax)
        ax.view_init(elev=24, azim=-58)
    fig.colorbar(s1, ax=ax1, shrink=0.62, pad=0.08)
    fig.colorbar(s2, ax=ax2, shrink=0.62, pad=0.08)
    subtitle = (
        f"{title_prefix} | {target}: "
        f"R²={safe_float(metrics.get('r2')):.5f}, "
        f"NMAE={safe_float(metrics.get('nmae')):.5f}, "
        f"NRMSE={safe_float(metrics.get('nrmse')):.5f}, "
        f"MAE={safe_float(metrics.get('mae')):.5g}, "
        f"RMSE={safe_float(metrics.get('rmse')):.5g}"
    )
    fig.suptitle(subtitle, fontsize=12, fontname="Times New Roman")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------


def run_selected_cycle_infer_plot(args: Any) -> Dict[str, Any]:
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt, ckpt_path, candidate_summary = load_candidate_checkpoint(args.candidate_dir, args.candidate_summary, args.checkpoint)
    device = device_from_arg(args.device)
    model = build_model_from_checkpoint(ckpt, device)
    feature_names_ckpt = list(ckpt.get("feature_names") or [])
    target_slices = {str(k): (int(v[0]), int(v[1])) for k, v in dict(ckpt.get("target_slices") or {}).items()}
    protocol_vocab, branch_vocab = parse_vocabs_from_checkpoint_feature_names(feature_names_ckpt, int(ckpt.get("local_input_dim", 0)))
    if not protocol_vocab or not branch_vocab:
        raise ValueError("Could not parse protocol/branch vocab from checkpoint feature_names")

    records, manifest = load_split_records(args.split_manifest)
    record = find_record(records, str(args.batch), str(args.battery))
    sem_map = load_semantics_map(args.g0_profile_semantics_csv)
    sem = semantics_for_record(record, sem_map)
    metric_targets = list(args.metric_targets)
    if "all" in [x.lower() for x in metric_targets]:
        metric_targets = list(target_slices.keys())
    plot_targets = list(args.plot_targets)
    if "both" in [x.lower() for x in plot_targets]:
        plot_targets = ["cs_a", "cs_c"]
    for pt in plot_targets:
        if pt not in metric_targets:
            metric_targets.append(pt)

    data = load_selected_cycle_data(
        record,
        sem,
        args.cycles,
        protocol_vocab,
        branch_vocab,
        metric_targets=metric_targets,
        max_points_per_cycle=int(args.max_points_per_cycle),
        prefer_replay_observed=not bool(args.prefer_softlabel_observed),
    )
    X = np.asarray(data["X"], dtype=np.float32)
    if X.shape[1] != np.asarray(ckpt["x_mean"]).size:
        raise ValueError(f"Feature dimension mismatch: selected X={X.shape[1]} checkpoint={np.asarray(ckpt['x_mean']).size}")
    if list(data["feature_names"]) != feature_names_ckpt:
        mismatch = []
        for i, (a, b) in enumerate(zip(data["feature_names"], feature_names_ckpt)):
            if a != b:
                mismatch.append((i, a, b))
                break
        raise ValueError(f"Feature name mismatch; first mismatch={mismatch[:1]}; selected_dim={len(data['feature_names'])}; ckpt_dim={len(feature_names_ckpt)}")

    pred_full = predict_array(model, X, ckpt, device, batch_size=int(args.predict_batch_size))
    pred_by_target = slice_prediction_by_targets(pred_full, target_slices, metric_targets)
    rows = make_metrics_rows(data, pred_by_target, record)
    agg = aggregate_selected_metrics(rows)

    metrics_csv = out_dir / "D17_G6F_SELECTED_CYCLE_METRICS.csv"
    write_csv_rows(rows, metrics_csv)

    temp_npz_path = None
    if bool(args.save_temp_npz) or bool(args.keep_temp_npz):
        temp_npz_path = out_dir / "tmp_predictions" / f"D17_G6F_{record.get('canonical_cell_uid') or record.get('cell_uid')}_cycles_{str(args.cycles).replace(',', '_').replace('-', 'to')}_PRED.npz"
        save_temp_prediction_npz(temp_npz_path, data, pred_by_target, keep_true=True)

    plot_files: List[str] = []
    if bool(args.plot_3d) or bool(args.save_png):
        if args.backend:
            import matplotlib
            matplotlib.use(str(args.backend), force=True)
        title_prefix = f"{record.get('canonical_cell_uid') or record.get('cell_uid')} cycles {args.cycles}"
        global_metrics_by_target = {str(r.get("target")): r for r in rows if r.get("level") == "selected_range_global"}
        for target in plot_targets:
            if target not in pred_by_target or target not in data["targets"]:
                continue
            side = "a" if target.endswith("_a") else "c"
            save_path = None
            if bool(args.save_png):
                save_path = out_dir / "figures" / f"D17_G6F_{record.get('canonical_cell_uid') or record.get('cell_uid')}_cycles_{str(args.cycles).replace(',', '_').replace('-', 'to')}_{target}_3D.png"
                plot_files.append(str(save_path))
            plot_3d_pair(
                target,
                np.asarray(data["t"], dtype=np.float32),
                np.asarray(data["radial"][side], dtype=np.float32),
                pred_by_target[target],
                data["targets"][target],
                global_metrics_by_target.get(target, {}),
                title_prefix=title_prefix,
                save_path=save_path,
                show=bool(args.plot_3d),
                time_axis=str(args.time_axis),
                max_time_points=int(args.plot_max_time_points),
                pred_cmap=str(args.pred_cmap),
                true_cmap=str(args.true_cmap),
            )

    if temp_npz_path and bool(args.delete_temp_predictions) and not bool(args.keep_temp_npz):
        try:
            Path(temp_npz_path).unlink(missing_ok=True)
            temp_npz_path = None
        except Exception:
            pass

    mean_gate = float(args.r2_mean_gate)
    min_gate = float(args.r2_min_gate)
    mean_r2 = safe_float(agg.get("selected_target_r2_mean"))
    min_r2 = safe_float(agg.get("selected_target_r2_min"))
    full_training_recommendation = "NO_FULL_TRAINING_INDICATED_FOR_THIS_SELECTED_RANGE" if mean_r2 >= mean_gate and min_r2 >= min_gate else "SELECTED_RANGE_BELOW_GATE_REVIEW_BEFORE_FULL_TRAINING"
    if not math.isfinite(mean_r2) or not math.isfinite(min_r2):
        full_training_recommendation = "METRICS_INVALID_REVIEW_INPUTS"

    summary = {
        "protocol": "D17-G6F_SELECTED_CYCLE_ON_DEMAND_INFERENCE_AND_PLOTTING",
        "created_at_utc": utc_now(),
        "status": "PASS",
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "purpose": "On-demand selected-cycle inference, report-only metrics, optional 3D plotting, and optional temporary prediction export.",
        "candidate_checkpoint": str(ckpt_path),
        "candidate_summary": str(args.candidate_summary or ""),
        "candidate_protocol": candidate_summary.get("protocol"),
        "candidate_status": candidate_summary.get("status"),
        "candidate_g6_ready": candidate_summary.get("g6_ready"),
        "candidate_g3_ready": candidate_summary.get("g3_ready"),
        "candidate_warning": "This selected-cycle result is local evidence only; full all-cell/all-cycle success still requires cycle-wise streaming audit.",
        "split_manifest": str(args.split_manifest),
        "manifest_hash_sha256": manifest.get("manifest_hash_sha256"),
        "record": {
            "split": record.get("split"),
            "cell_uid": record.get("cell_uid"),
            "canonical_cell_uid": record.get("canonical_cell_uid"),
            "batch": record.get("batch"),
            "battery": record.get("battery"),
            "protocol": record.get("protocol"),
            "softlabel_npz": record.get("softlabel_npz"),
            "replay_npz": record.get("replay_npz"),
        },
        "semantic_branch": data.get("branch"),
        "requested_cycles": str(args.cycles),
        "evaluated_cycles": data.get("selected_cycles"),
        "n_time_points": int(np.asarray(data["t"]).size),
        "metric_targets": metric_targets,
        "plot_targets": plot_targets,
        "aggregate_metrics": agg,
        "r2_mean_gate": mean_gate,
        "r2_min_gate": min_gate,
        "full_training_recommendation": full_training_recommendation,
        "observed_sources": data.get("observed_sources"),
        "files": {
            "summary_json": str(out_dir / "D17_G6F_SELECTED_CYCLE_SUMMARY.json"),
            "metrics_csv": str(metrics_csv),
            "temp_prediction_npz": str(temp_npz_path) if temp_npz_path else "",
            "plot_files": plot_files,
        },
        "elapsed_s": time.time() - t0,
    }
    write_json(summary, out_dir / "D17_G6F_SELECTED_CYCLE_SUMMARY.json")
    return summary
