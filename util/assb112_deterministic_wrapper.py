# -*- coding: utf-8 -*-
"""Deterministic ModelFin_112 wrapper helpers.

This module intentionally avoids the neural SOH head.  It loads the deterministic
ridge SOH head produced by ``train_assb112_deterministic_soh_baseline.py`` and
combines it with frozen ModelFin_107A state outputs through a single wrapper
manifest.  The wrapper is an engineering single-model interface, not an
end-to-end jointly trained PINN.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import json
import math

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

STATE_VARIABLES: Tuple[str, ...] = ("cs_a", "cs_c", "phie", "phis_c")
SOH_VARIABLE = "SOH"


def json_clean(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_clean(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if not math.isfinite(v) else v
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    return x


def load_json(path: PathLike) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Mapping[str, Any], path: PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(json_clean(dict(obj)), f, ensure_ascii=False, indent=2, sort_keys=True)


def resolve_path(path: PathLike, *, base_dir: Optional[PathLike] = None, root_dir: Optional[PathLike] = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    candidates: List[Path] = []
    if base_dir is not None:
        candidates.append(Path(base_dir) / p)
    if root_dir is not None:
        candidates.append(Path(root_dir) / p)
    candidates.append(Path.cwd() / p)
    for c in candidates:
        if c.exists():
            return c
    # Return the most semantically useful unresolved candidate.
    if base_dir is not None:
        return Path(base_dir) / p
    if root_dir is not None:
        return Path(root_dir) / p
    return p


def metrics(y_true: Any, y_pred: Any) -> Dict[str, float]:
    a = np.asarray(y_true, dtype=float).reshape(-1)
    b = np.asarray(y_pred, dtype=float).reshape(-1)
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]
    if a.size == 0:
        return {"n": 0, "MAE": math.nan, "RMSE": math.nan, "NMAE": math.nan, "NRMSE": math.nan, "R2": math.nan, "corr": math.nan, "BIAS": math.nan}
    err = b - a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    bias = float(np.mean(err))
    denom = float(np.sum((a - np.mean(a)) ** 2))
    r2 = float(1.0 - np.sum(err * err) / denom) if denom > 1.0e-15 else math.nan
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size >= 2 and np.std(a) > 1.0e-15 and np.std(b) > 1.0e-15 else math.nan
    rng = float(np.max(a) - np.min(a)) if a.size else math.nan
    nmae = float(mae / rng) if rng > 1.0e-12 else math.nan
    nrmse = float(rmse / rng) if rng > 1.0e-12 else math.nan
    return {"n": int(a.size), "MAE": mae, "RMSE": rmse, "NMAE": nmae, "NRMSE": nrmse, "R2": r2, "corr": corr, "BIAS": bias}


def soh_metrics_to_score_row(split: str, m: Mapping[str, Any], source: str) -> Dict[str, Any]:
    return {
        "variable": "SOH",
        "source": source,
        "split": split,
        "n": m.get("n"),
        "MAE": m.get("SOH_MAE", m.get("MAE")),
        "RMSE": m.get("SOH_RMSE", m.get("RMSE")),
        "NMAE": m.get("SOH_NMAE", m.get("NMAE")),
        "NRMSE": m.get("SOH_NRMSE", m.get("NRMSE")),
        "R2": m.get("SOH_R2", m.get("R2")),
        "corr": m.get("SOH_corr", m.get("corr")),
        "BIAS": m.get("SOH_BIAS", m.get("BIAS")),
    }


def load_scaler(path: PathLike) -> Dict[str, Any]:
    return load_json(path)


def transform_with_scaler(frame: pd.DataFrame, scaler: Mapping[str, Any]) -> np.ndarray:
    cols = list(scaler.get("feature_columns", []))
    if not cols:
        raise KeyError("feature_scaler.json has no feature_columns")
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"dataset_csv is missing deterministic SOH feature columns: {missing}")
    x = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    med = np.asarray(scaler.get("median_impute", np.zeros(len(cols))), dtype=float)
    if med.size != len(cols):
        med = np.zeros(len(cols), dtype=float)
    bad = np.where(~np.isfinite(x))
    if bad[0].size:
        x[bad] = np.take(med, bad[1])
    mean = np.asarray(scaler.get("mean", np.zeros(len(cols))), dtype=float)
    std = np.asarray(scaler.get("std", np.ones(len(cols))), dtype=float)
    if mean.size != len(cols):
        mean = np.zeros(len(cols), dtype=float)
    if std.size != len(cols):
        std = np.ones(len(cols), dtype=float)
    std[~np.isfinite(std) | (std < 1.0e-12)] = 1.0
    return (x - mean) / std


class DeterministicRidgeSOH:
    def __init__(self, model_dir: PathLike):
        self.model_dir = Path(model_dir)
        model_path = self.model_dir / "deterministic_soh_model.json"
        if not model_path.exists():
            model_path = self.model_dir / "ridge_model.json"
        self.model = load_json(model_path)
        self.scaler = load_scaler(self.model_dir / str(self.model.get("scaler_json", "feature_scaler.json")))
        self.feature_columns = list(self.model.get("feature_columns") or self.scaler.get("feature_columns") or [])
        if not self.feature_columns:
            raise RuntimeError(f"No feature columns found in deterministic SOH model: {model_path}")
        self.coef = np.asarray(self.model.get("coef_intercept_first", []), dtype=float)
        if self.coef.size != len(self.feature_columns) + 1:
            raise RuntimeError(
                f"coef_intercept_first length mismatch: got {self.coef.size}, expected {len(self.feature_columns)+1}"
            )
        self.clip_min = float(self.model.get("clip_soh_min", 0.0))
        self.clip_max = float(self.model.get("clip_soh_max", 1.05))

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = transform_with_scaler(frame, self.scaler)
        pred = np.column_stack([np.ones(x.shape[0], dtype=float), x]) @ self.coef
        pred = np.clip(pred, self.clip_min, self.clip_max)
        out = pd.DataFrame()
        if "cycle_id" in frame.columns:
            out["cycle_id"] = pd.to_numeric(frame["cycle_id"], errors="coerce").astype("Int64")
        else:
            out["cycle_id"] = np.arange(len(frame), dtype=int)
        if "split" in frame.columns:
            out["split"] = frame["split"].astype(str).to_numpy()
        out["SOH_pred"] = pred.astype(float)
        if "SOH_obs" in frame.columns:
            out["SOH_obs"] = pd.to_numeric(frame["SOH_obs"], errors="coerce").to_numpy(dtype=float)
            out["SOH_err"] = out["SOH_pred"] - out["SOH_obs"]
        return out


def load_deterministic_soh_from_wrapper(model_dir: PathLike) -> DeterministicRidgeSOH:
    md = Path(model_dir)
    cfg_path = md / "unified_config.json"
    if cfg_path.exists():
        cfg = load_json(cfg_path)
        soh_ref = cfg.get("soh_model_dir", cfg.get("soh_head_dir", "soh_deterministic"))
        soh_dir = resolve_path(soh_ref, base_dir=md, root_dir=Path.cwd())
    else:
        soh_dir = md
    return DeterministicRidgeSOH(soh_dir)


def find_npz_key(data: Mapping[str, np.ndarray], var: str, kind: str) -> Optional[str]:
    keys = list(data.keys())
    low_map = {k.lower(): k for k in keys}
    kind_tokens = {
        "true": ["true", "label", "target", "obs", "gt", "ref", "y"],
        "pred": ["pred", "hat", "estimate", "corrected"],
    }[kind]
    candidates: List[str] = []
    for sep in ["_", "", "-"]:
        for token in kind_tokens:
            if kind == "true":
                candidates.extend([f"{var}{sep}{token}", f"{token}{sep}{var}"])
            else:
                candidates.extend([f"{var}{sep}{token}", f"{token}{sep}{var}"])
    if var == "phis_c":
        alt = ["phis", "phi_s_c", "phis_ca", "phi_s_ca"]
        for a in alt:
            for token in kind_tokens:
                candidates.extend([f"{a}_{token}", f"{token}_{a}"])
    if var == "phie":
        alt = ["phi_e", "phie"]
        for a in alt:
            for token in kind_tokens:
                candidates.extend([f"{a}_{token}", f"{token}_{a}"])
    for c in candidates:
        if c.lower() in low_map:
            return low_map[c.lower()]
    # Fuzzy fallback: require var tokens and one kind token.
    var_tokens = var.lower().split("_")
    for k in keys:
        kl = k.lower()
        if all(t in kl for t in var_tokens) and any(tok in kl for tok in kind_tokens):
            return k
    return None


def load_state_metrics_from_npz(npz_path: PathLike) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    p = Path(npz_path)
    data = np.load(p, allow_pickle=True)
    arrs = {k: data[k] for k in data.files}
    rows: List[Dict[str, Any]] = []
    audit: Dict[str, Any] = {"state_eval_npz": str(p), "missing": [], "used_keys": {}}
    for var in STATE_VARIABLES:
        tk = find_npz_key(arrs, var, "true")
        pk = find_npz_key(arrs, var, "pred")
        if tk is None or pk is None:
            audit["missing"].append({"variable": var, "true_key": tk, "pred_key": pk})
            continue
        row = {"variable": var, "source": f"state_eval_npz:{p.name}", **metrics(arrs[tk], arrs[pk])}
        rows.append(row)
        audit["used_keys"][var] = {"true": tk, "pred": pk}
    return rows, audit


def load_state_rows_from_scorecard(scorecard_csv: PathLike) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    p = Path(scorecard_csv)
    df = pd.read_csv(p)
    if "variable" not in df.columns:
        # common historical fallback
        for c in df.columns:
            if c.lower() in {"var", "target", "name"}:
                df = df.rename(columns={c: "variable"})
                break
    if "variable" not in df.columns:
        raise KeyError(f"state scorecard has no variable column: {p}")
    out = df[df["variable"].astype(str).isin(STATE_VARIABLES)].copy()
    if "source" not in out.columns:
        out["source"] = f"state_scorecard:{p.name}"
    rows = out.to_dict(orient="records")
    return rows, {"state_scorecard_csv": str(p), "n_state_rows": len(rows)}


def default_state_scorecard_candidates(root: PathLike = ".") -> List[Path]:
    root = Path(root)
    explicit = [
        root / "EvalFin_112_deterministic_wrapper" / "state_scorecard.csv",
        root / "EvalFin_111_seed42locked_repro_c00" / "five_state_scorecard.csv",
        root / "EvalFin_110_joint_StageB_SOH_107A_states" / "five_state_scorecard.csv",
        root / "EvalFin_110_joint_StageB_SOH_107A_states_fix2" / "five_state_scorecard.csv",
    ]
    globbed = list(root.glob("EvalFin_107A*/five_state_scorecard.csv"))
    globbed += list(root.glob("EvalFin_107A*/metrics_global.csv"))
    return [p for p in explicit + globbed if p.exists()]


def default_state_npz_candidates(root: PathLike = ".") -> List[Path]:
    root = Path(root)
    names = ["evaluation_paired.npz", "paired_eval.npz", "eval_paired.npz", "paired_predictions.npz", "evaluation_arrays.npz"]
    dirs = [root / "EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only"]
    dirs += list(root.glob("EvalFin_107A*"))
    candidates: List[Path] = []
    for d in dirs:
        for n in names:
            p = d / n
            if p.exists():
                candidates.append(p)
        candidates += list(d.glob("*.npz")) if d.exists() else []
    # de-duplicate while preserving order
    seen = set()
    out = []
    for p in candidates:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def soh_score_rows_from_prediction(pred_frame: pd.DataFrame) -> List[Dict[str, Any]]:
    if "SOH_obs" not in pred_frame.columns:
        return []
    rows: List[Dict[str, Any]] = []
    if "split" in pred_frame.columns:
        split_series = pred_frame["split"].astype(str).str.lower()
    else:
        split_series = pd.Series(["all"] * len(pred_frame), index=pred_frame.index)
    for split in ["train", "val", "test", "partial"]:
        mask = split_series.eq(split)
        if bool(mask.any()):
            rows.append({"variable": "SOH", "source": "deterministic_ridge_wrapper", "split": split, **metrics(pred_frame.loc[mask, "SOH_obs"], pred_frame.loc[mask, "SOH_pred"])})
    eval_mask = split_series.isin(["train", "val", "test"])
    if bool(eval_mask.any()):
        rows.append({"variable": "SOH", "source": "deterministic_ridge_wrapper", "split": "all_eval", **metrics(pred_frame.loc[eval_mask, "SOH_obs"], pred_frame.loc[eval_mask, "SOH_pred"])})
    rows.append({"variable": "SOH", "source": "deterministic_ridge_wrapper", "split": "all_rows", **metrics(pred_frame["SOH_obs"], pred_frame["SOH_pred"])})
    return rows


def write_scorecard(rows: Sequence[Mapping[str, Any]], path: PathLike) -> pd.DataFrame:
    df = pd.DataFrame([dict(r) for r in rows])
    # Stable column ordering; keep any extra columns after the main ones.
    main = ["variable", "source", "split", "n", "MAE", "RMSE", "NMAE", "NRMSE", "R2", "corr", "BIAS"]
    cols = [c for c in main if c in df.columns] + [c for c in df.columns if c not in main]
    df = df[cols] if cols else df
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return df
