# -*- coding: utf-8 -*-
"""Plot ASSB-111 SOH prediction by cycle.

This script reads ``soh_pred_by_cycle.csv`` and writes a static PNG. If Plotly is
available, it also writes an interactive HTML figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


def _as_float(s):
    return pd.to_numeric(pd.Series(s), errors="coerce").to_numpy(dtype=np.float64)


def _maybe_cols(df: pd.DataFrame, names: Sequence[str]) -> List[str]:
    return [c for c in names if c in df.columns]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot ASSB111 SOH predictions")
    p.add_argument("--pred_csv", default=r"EvalFin_111_saturating_v2_strict30_test70\soh_pred_by_cycle.csv")
    p.add_argument("--output_dir", default="", help="Defaults to parent directory of --pred_csv")
    p.add_argument("--output_png", default="")
    p.add_argument("--output_html", default="")
    p.add_argument("--cycle_col", default="cycle_id")
    p.add_argument("--split_col", default="split")
    p.add_argument("--obs_col", default="SOH_obs")
    p.add_argument("--pred_col", default="SOH_pred")
    p.add_argument("--title", default="ASSB-111 strict30 SOH prediction")
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--no_html", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    pred_csv = Path(args.pred_csv)
    if not pred_csv.exists():
        raise FileNotFoundError(pred_csv)
    out_dir = Path(args.output_dir) if args.output_dir else pred_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    output_png = Path(args.output_png) if args.output_png else out_dir / "soh_prediction.png"
    output_html = Path(args.output_html) if args.output_html else out_dir / "soh_prediction.html"

    df = pd.read_csv(pred_csv).copy()
    missing = [c for c in [args.cycle_col, args.split_col, args.pred_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {pred_csv}: {missing}")
    df[args.cycle_col] = pd.to_numeric(df[args.cycle_col], errors="coerce")
    df = df[np.isfinite(df[args.cycle_col])].sort_values(args.cycle_col).reset_index(drop=True)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = _as_float(df[args.cycle_col])
    if args.obs_col in df.columns:
        y_obs = _as_float(df[args.obs_col])
        ax.plot(x, y_obs, marker=".", linewidth=1.0, markersize=3.0, label=args.obs_col)
    y_pred = _as_float(df[args.pred_col])
    ax.plot(x, y_pred, marker=".", linewidth=1.0, markersize=3.0, label=args.pred_col)

    extra_cols = _maybe_cols(df, ["SOH_struct", "SOH_base", "SOH_pred_unclipped"])
    for col in extra_cols:
        ax.plot(x, _as_float(df[col]), linewidth=0.9, linestyle="--", label=col)

    # Mark split boundaries using neutral vertical lines. No training logic uses these.
    if args.split_col in df.columns:
        for split in ["train", "val", "test", "partial"]:
            sub = df[df[args.split_col].astype(str) == split]
            if len(sub):
                xmin = float(np.nanmin(_as_float(sub[args.cycle_col])))
                xmax = float(np.nanmax(_as_float(sub[args.cycle_col])))
                ax.axvline(xmin, linewidth=0.8, linestyle=":")
                ax.text(xmin, ax.get_ylim()[1], split, rotation=90, va="top", ha="right", fontsize=8)
                if split == "test":
                    ax.axvline(xmax, linewidth=0.8, linestyle=":")
    ax.set_xlabel("cycle_id")
    ax.set_ylabel("SOH")
    ax.set_title(args.title)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=int(args.dpi))
    plt.close(fig)

    html_written = False
    if not args.no_html:
        try:
            import plotly.graph_objects as go
            fig_i = go.Figure()
            if args.obs_col in df.columns:
                fig_i.add_trace(go.Scatter(x=x, y=_as_float(df[args.obs_col]), mode="lines+markers", name=args.obs_col))
            fig_i.add_trace(go.Scatter(x=x, y=y_pred, mode="lines+markers", name=args.pred_col))
            for col in extra_cols:
                fig_i.add_trace(go.Scatter(x=x, y=_as_float(df[col]), mode="lines", name=col))
            fig_i.update_layout(title=args.title, xaxis_title="cycle_id", yaxis_title="SOH")
            output_html.parent.mkdir(parents=True, exist_ok=True)
            fig_i.write_html(str(output_html), include_plotlyjs="cdn")
            html_written = True
        except Exception as exc:
            # Keep PNG path usable even when plotly is absent.
            (out_dir / "soh_prediction_html_error.json").write_text(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "pred_csv": str(pred_csv),
        "output_png": str(output_png),
        "output_html": str(output_html) if html_written else None,
        "n_rows": int(len(df)),
        "columns_used": [args.cycle_col, args.split_col, args.pred_col] + ([args.obs_col] if args.obs_col in df.columns else []) + extra_cols,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
