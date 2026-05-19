# -*- coding: utf-8 -*-
"""Feature-group audit for ASSB-111/112 strict30 SOH prediction.

The audit deliberately uses a simple ridge baseline.  It fits only train cycles,
selects the ridge strength using visible validation cycles, and reports test
metrics only after the model is fixed.  This gives a fast, no-GPU answer to the
question: do G1/G2/G3/G4 features carry information beyond cycle/throughput G0?
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

try:
    from util.assb_soh_feature_schema import (
        audit_feature_frame,
        feature_group_for_column,
        fit_standard_scaler,
        select_feature_columns,
        transform_with_scaler,
        write_scaler_json,
        write_schema_json,
        _json_clean,  # type: ignore
    )
except Exception:  # pragma: no cover
    from assb_soh_feature_schema import audit_feature_frame, feature_group_for_column, fit_standard_scaler, select_feature_columns, transform_with_scaler, write_scaler_json, write_schema_json, _json_clean  # type: ignore


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Audit ASSB strict30 feature groups")
    p.add_argument("--dataset_csv", required=True)
    p.add_argument("--split_manifest_json", default="", help="Optional strict30 manifest JSON; default uses train=5-139/val=140-159/test=160-521")
    p.add_argument("--output_dir", default=r"EvalFin_112_feature_audit_v1")
    p.add_argument("--groups", default="G0,G1,G2,G3,G4")
    p.add_argument("--alphas", default="1e-6,1e-4,1e-3,1e-2,1e-1,1,10")
    p.add_argument("--seeds", default="7,42,2026,3407,7890", help="Bootstrap seeds for train-only stability check; use empty for deterministic")
    p.add_argument("--bootstrap_fraction", type=float, default=1.0, help="Train-only bootstrap fraction; 1.0 keeps all train rows")
    p.add_argument("--allow_upper_bound", action="store_true")
    p.add_argument("--target_col", default="SOH_obs")
    return p.parse_args(argv)


def _save_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(dict(obj)), f, ensure_ascii=False, indent=2, sort_keys=True)


def _parse_floats(s: str) -> List[float]:
    return [float(x) for x in str(s).replace(";", ",").split(",") if str(x).strip()]


def _parse_ints(s: str) -> List[int]:
    if not str(s).strip():
        return []
    return [int(float(x)) for x in str(s).replace(";", ",").split(",") if str(x).strip()]


def _load_manifest(path: str) -> Dict[str, Any]:
    if path and Path(path).exists():
        with Path(path).open("r", encoding="utf-8") as f:
            m = json.load(f)
    else:
        m = {}
    # Fill defaults if keys are missing.
    m.setdefault("train_cycle_from", 5)
    m.setdefault("train_cycle_to", 139)
    m.setdefault("val_cycle_from", 140)
    m.setdefault("val_cycle_to", 159)
    m.setdefault("test_cycle_from", 160)
    m.setdefault("test_cycle_to", 521)
    m.setdefault("partial_cycles", [522])
    return m


def _split_from_manifest(cycles: Sequence[int], manifest: Mapping[str, Any]) -> np.ndarray:
    c = np.asarray(cycles, dtype=int)
    out = np.full(c.shape, "out_of_scope", dtype=object)

    def _range_mask(prefix: str) -> np.ndarray:
        a = manifest.get(f"{prefix}_cycle_from", None)
        b = manifest.get(f"{prefix}_cycle_to", None)
        if a is None or b is None:
            vals = manifest.get(f"{prefix}_cycles", [])
            return np.isin(c, [int(x) for x in vals])
        return (c >= int(a)) & (c <= int(b))

    out[_range_mask("train")] = "train"
    out[_range_mask("val")] = "val"
    out[_range_mask("test")] = "test"
    partial = manifest.get("partial_cycles", []) or manifest.get("partial", [])
    if isinstance(partial, (int, float, str)):
        partial = [partial]
    out[np.isin(c, [int(float(x)) for x in partial])] = "partial"
    return out


def _metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(pred)
    if int(np.sum(m)) == 0:
        return {"n": 0, "MAE": math.nan, "RMSE": math.nan, "R2": math.nan, "corr": math.nan, "BIAS": math.nan}
    yy = y[m]
    pp = pred[m]
    err = pp - yy
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    denom = float(np.sum((yy - np.mean(yy)) ** 2))
    r2 = 1.0 - float(np.sum(err * err)) / denom if denom > 1e-12 else math.nan
    corr = float(np.corrcoef(yy, pp)[0, 1]) if len(yy) > 1 and np.std(yy) > 1e-12 and np.std(pp) > 1e-12 else math.nan
    return {"n": int(len(yy)), "MAE": mae, "RMSE": rmse, "R2": r2, "corr": corr, "BIAS": float(np.mean(err))}


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ones = np.ones((X.shape[0], 1), dtype=float)
    Xb = np.hstack([ones, X])
    reg = np.eye(Xb.shape[1], dtype=float) * float(alpha)
    reg[0, 0] = 0.0  # do not penalize intercept
    return np.linalg.pinv(Xb.T @ Xb + reg) @ (Xb.T @ y)


def _ridge_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1), dtype=float), X]) @ beta


def _fit_select_predict(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    y: np.ndarray,
    split: np.ndarray,
    alphas: Sequence[float],
    *,
    seed: Optional[int],
    bootstrap_fraction: float,
) -> Tuple[np.ndarray, Dict[str, Any], pd.DataFrame]:
    train_mask = split == "train"
    val_mask = split == "val"
    if not np.any(train_mask):
        raise RuntimeError("No train cycles in feature audit")
    fit_idx = np.where(train_mask)[0]
    if seed is not None and 0.0 < bootstrap_fraction < 1.0:
        rng = np.random.default_rng(int(seed))
        n_pick = max(5, int(round(len(fit_idx) * float(bootstrap_fraction))))
        fit_idx = rng.choice(fit_idx, size=min(n_pick, len(fit_idx)), replace=False)
        train_fit_mask = np.zeros(len(frame), dtype=bool)
        train_fit_mask[fit_idx] = True
    else:
        train_fit_mask = train_mask

    scaler = fit_standard_scaler(frame, feature_columns, fit_mask=train_fit_mask)
    X = transform_with_scaler(frame, scaler)
    rows: List[Dict[str, Any]] = []
    best_alpha = None
    best_score = float("inf")
    best_beta = None
    for alpha in alphas:
        beta = _ridge_fit(X[train_fit_mask], y[train_fit_mask], alpha)
        pred = _ridge_predict(X, beta)
        tr = _metrics(y[train_mask], pred[train_mask])
        va = _metrics(y[val_mask], pred[val_mask])
        score = va["MAE"] if np.isfinite(va["MAE"]) else tr["MAE"]
        rows.append({"alpha": float(alpha), "visible_score_val_mae": score, "train_MAE": tr["MAE"], "train_R2": tr["R2"], "val_MAE": va["MAE"], "val_R2": va["R2"]})
        if np.isfinite(score) and score < best_score:
            best_score = float(score)
            best_alpha = float(alpha)
            best_beta = beta
    if best_beta is None:
        raise RuntimeError("Failed to select ridge alpha")
    pred = _ridge_predict(X, best_beta)
    return pred, {"best_alpha": best_alpha, "scaler": scaler, "beta": best_beta.tolist(), "visible_score": best_score}, pd.DataFrame(rows)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.dataset_csv)
    if "cycle_id" not in frame.columns:
        raise KeyError("dataset_csv must contain cycle_id")
    if args.target_col not in frame.columns:
        raise KeyError(f"dataset_csv must contain target column {args.target_col!r}")
    frame["cycle_id"] = frame["cycle_id"].astype(int)
    manifest = _load_manifest(args.split_manifest_json)
    split = frame["split"].astype(str).str.lower().to_numpy() if "split" in frame.columns else _split_from_manifest(frame["cycle_id"].to_numpy(), manifest)
    frame["split"] = split
    y = pd.to_numeric(frame[args.target_col], errors="coerce").to_numpy(dtype=float)
    alphas = _parse_floats(args.alphas)
    seeds = _parse_ints(args.seeds)
    if not seeds:
        seeds = [None]  # type: ignore[list-item]
    groups = [g.strip() for g in str(args.groups).replace(";", ",").split(",") if g.strip()]

    all_metric_rows: List[Dict[str, Any]] = []
    all_alpha_rows: List[Dict[str, Any]] = []
    all_importance_rows: List[Dict[str, Any]] = []
    group_summaries: Dict[str, Any] = {}
    failures: List[str] = []
    warnings: List[str] = []

    for group in groups:
        try:
            cols = select_feature_columns(frame, group, allow_upper_bound=bool(args.allow_upper_bound), allow_missing=False)
        except Exception as exc:
            failures.append(f"{group}: cannot select features: {exc}")
            continue
        audit = audit_feature_frame(frame, cols, allow_upper_bound=bool(args.allow_upper_bound))
        if not audit["ok"]:
            failures.extend([f"{group}: {x}" for x in audit["failures"]])
            continue
        warnings.extend([f"{group}: {x}" for x in audit.get("warnings", [])])
        write_schema_json(out_dir / f"feature_schema_{group}.json", group, allow_upper_bound=bool(args.allow_upper_bound))
        preds_by_seed: List[np.ndarray] = []
        selected_rows: List[Dict[str, Any]] = []
        for seed in seeds:
            pred, selection, alpha_df = _fit_select_predict(
                frame,
                cols,
                y,
                split,
                alphas,
                seed=seed,
                bootstrap_fraction=float(args.bootstrap_fraction),
            )
            preds_by_seed.append(pred)
            seed_label = "deterministic" if seed is None else str(seed)
            alpha_df.insert(0, "group", group)
            alpha_df.insert(1, "seed", seed_label)
            all_alpha_rows.extend(alpha_df.to_dict(orient="records"))
            selected_rows.append({"seed": seed_label, "best_alpha": selection["best_alpha"], "visible_score": selection["visible_score"]})
            for split_name in ["train", "val", "test", "partial", "visible"]:
                if split_name == "visible":
                    mask = np.isin(split, ["train", "val"])
                else:
                    mask = split == split_name
                m = _metrics(y[mask], pred[mask])
                all_metric_rows.append({"group": group, "seed": seed_label, "split": split_name, **m})
            # Ridge coefficient importance proxy.
            beta = np.asarray(selection["beta"], dtype=float)[1:]
            for c, b in zip(cols, beta):
                all_importance_rows.append({
                    "group": group,
                    "seed": seed_label,
                    "feature": c,
                    "feature_group": feature_group_for_column(c),
                    "importance_abs_beta": float(abs(b)),
                    "signed_beta": float(b),
                })
        pred_mean = np.mean(np.stack(preds_by_seed, axis=0), axis=0)
        for split_name in ["train", "val", "test", "partial", "visible"]:
            mask = np.isin(split, ["train", "val"]) if split_name == "visible" else split == split_name
            all_metric_rows.append({"group": group, "seed": "mean_prediction", "split": split_name, **_metrics(y[mask], pred_mean[mask])})
        group_summaries[group] = {
            "n_features": len(cols),
            "features": list(cols),
            "selected_by_seed": selected_rows,
            "feature_audit": audit,
            "selection_uses_test": False,
        }

    metrics_df = pd.DataFrame(all_metric_rows)
    alpha_df = pd.DataFrame(all_alpha_rows)
    imp_df = pd.DataFrame(all_importance_rows)
    metrics_df.to_csv(out_dir / "feature_group_metrics.csv", index=False, encoding="utf-8-sig")
    alpha_df.to_csv(out_dir / "ridge_alpha_selection_visible_only.csv", index=False, encoding="utf-8-sig")
    imp_df.to_csv(out_dir / "feature_importance_by_group.csv", index=False, encoding="utf-8-sig")

    # Summary table: mean_prediction test metrics and seed statistics.
    summary_rows: List[Dict[str, Any]] = []
    if not metrics_df.empty:
        for group in groups:
            gdf = metrics_df[(metrics_df["group"] == group) & (metrics_df["split"] == "test")]
            seed_df = gdf[gdf["seed"] != "mean_prediction"]
            mean_df = gdf[gdf["seed"] == "mean_prediction"]
            row: Dict[str, Any] = {"group": group}
            if not mean_df.empty:
                r = mean_df.iloc[0]
                row.update({"test_R2_mean_prediction": r["R2"], "test_MAE_mean_prediction": r["MAE"], "test_RMSE_mean_prediction": r["RMSE"], "test_corr_mean_prediction": r["corr"]})
            if not seed_df.empty:
                row.update({
                    "test_R2_seed_mean": float(seed_df["R2"].mean()),
                    "test_R2_seed_min": float(seed_df["R2"].min()),
                    "test_MAE_seed_mean": float(seed_df["MAE"].mean()),
                    "test_MAE_seed_max": float(seed_df["MAE"].max()),
                })
            summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out_dir / "feature_group_summary.csv", index=False, encoding="utf-8-sig")

    # Dominance check: are the top importances only cycle/time features?
    dominance: Dict[str, Any] = {}
    if not imp_df.empty:
        for group in groups:
            sub = imp_df[(imp_df["group"] == group) & (imp_df["seed"] != "mean_prediction")].copy()
            if sub.empty:
                continue
            agg = sub.groupby(["feature", "feature_group"], as_index=False)["importance_abs_beta"].mean().sort_values("importance_abs_beta", ascending=False)
            top = agg.head(10)
            dominance[group] = {
                "top10_features": top.to_dict(orient="records"),
                "top10_G0_count": int(top["feature_group"].astype(str).str.contains("G0", case=False).sum()),
            }
    ok = len(failures) == 0
    _save_json(
        {
            "ok": ok,
            "dataset_csv": str(args.dataset_csv),
            "split_manifest_json": str(args.split_manifest_json),
            "groups": groups,
            "seeds": seeds,
            "fit_splits": ["train"],
            "selection_splits": ["val"],
            "test_metrics_used_for_selection": False,
            "failures": failures,
            "warnings": warnings,
            "group_summaries": group_summaries,
            "importance_dominance": dominance,
            "strict_note": "Ridge models are fit on train only; alpha is selected by val only; test metrics are reported after selection.",
        },
        out_dir / "feature_audit_summary.json",
    )
    print(f"[OK] wrote audit to {out_dir}")
    if failures:
        print("[FAIL] " + "; ".join(failures))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
