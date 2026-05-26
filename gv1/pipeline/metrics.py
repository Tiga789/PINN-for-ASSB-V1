from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred) -> dict[str, float | int | None]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return {'n': 0, 'mae': None, 'rmse': None, 'bias': None, 'corr': None, 'r2': None, 'nmae': None, 'nrmse': None}
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = len(y_true)
    if n == 0:
        return {'n': 0, 'mae': None, 'rmse': None, 'bias': None, 'corr': None, 'r2': None, 'nmae': None, 'nrmse': None}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    var = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / var) if var > 0 else None
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if n > 2 and np.std(y_true) > 0 and np.std(y_pred) > 0 else None
    span = float(np.nanmax(y_true) - np.nanmin(y_true)) if n else 0.0
    return {
        'n': int(n),
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'corr': corr,
        'r2': r2,
        'nmae': float(mae / span) if span > 0 else None,
        'nrmse': float(rmse / span) if span > 0 else None,
    }


def write_metrics_json(metrics: Mapping, output_json: str | Path) -> None:
    p = Path(output_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(metrics), ensure_ascii=False, indent=2), encoding='utf-8')


def summarize_scorecard(rows: list[dict], output_csv: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    p = Path(output_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding='utf-8-sig')
    return df
