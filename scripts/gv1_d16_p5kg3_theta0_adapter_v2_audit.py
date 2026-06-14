#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D16-P5K-G3 observed-only theta0 adapter v2 diagnostic audit.

No training, no checkpoint loading, no model mutation.
This script reads the G1 by-profile audit CSV and diagnoses whether a stronger
observed/hard-regime-gated theta0 adapter can plausibly replace the failed G1 rule_v1.

Important boundaries:
- Exact state metrics are only available for candidates already materialized in G1
  (P5K-C-baseline, G1-rule_v1, G1-ridge_*, G1-theta0_oracle).
- New hand-crafted rule_v2 candidates are evaluated against oracle initial shifts at
  profile level. They are NOT promoted as exact state-metric candidates unless a later
  array-level audit recomputes exact metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_CACHE_ROOT = Path(r"E:\XJTU battery dataset\_gv1_cache")
DEFAULT_OUT_DIR = DEFAULT_CACHE_ROOT / "xjtu_d16_p5kg3_theta0_adapter_v2_audit"

G1_AUTO_PATHS = [
    DEFAULT_CACHE_ROOT / "xjtu_d16_p5kg1_MINI_EVIDENCE" / "D16_P5KG1_OBSERVED_THETA0_BY_PROFILE.csv",
    DEFAULT_CACHE_ROOT / "xjtu_d16_p5kg1_observed_theta0_audit" / "D16_P5KG1_OBSERVED_THETA0_BY_PROFILE.csv",
]

G0_AUTO_PATHS = [
    DEFAULT_CACHE_ROOT / "xjtu_d16_p5kg0_MINI_EVIDENCE" / "D16_P5KG0_BASELINE_REPAIR_BY_PROFILE.csv",
    DEFAULT_CACHE_ROOT / "xjtu_d16_p5kg0_baseline_repair_audit" / "D16_P5KG0_BASELINE_REPAIR_BY_PROFILE.csv",
    DEFAULT_CACHE_ROOT / "xjtu_d16_p5kg0_baseline_repair_audit" / "D16_P5KG0_BASELINE_REPAIR_BY_PROFILE.csv",
]

REPORT_NAME = "D16_P5KG3_THETA0_ADAPTER_V2_AUDIT_REPORT.md"
SPLIT_NAME = "D16_P5KG3_THETA0_ADAPTER_V2_SPLIT_SUMMARY.csv"
PROFILE_NAME = "D16_P5KG3_THETA0_ADAPTER_V2_BY_PROFILE.csv"
CAND_NAME = "D16_P5KG3_THETA0_ADAPTER_V2_CANDIDATE_SUMMARY.csv"
FAIL_NAME = "D16_P5KG3_THETA0_ADAPTER_V2_FAILURES.json"
SUMMARY_NAME = "D16_P5KG3_THETA0_ADAPTER_V2_SUMMARY.json"

METRIC_COLS = [
    "theta_a_mean_mae", "theta_a_mean_bias", "theta_a_mean_r2",
    "theta_c_mean_mae", "theta_c_mean_bias", "theta_c_mean_r2",
    "cs_a_mean_r2", "cs_c_mean_r2",
]

REFERENCE_BASELINE = {
    "theta_a_mean_mae": 0.139017,
    "theta_a_mean_r2": 0.474238,
    "theta_c_mean_mae": 0.123569,
    "theta_c_mean_r2": 0.391913,
}


def norm_path(p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    return Path(p)


def find_g1_by_profile(user_path: Optional[str]) -> Path:
    if user_path:
        p = Path(user_path)
        if not p.exists():
            raise FileNotFoundError(f"G1 by-profile CSV not found: {p}")
        return p
    for p in G1_AUTO_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot locate D16_P5KG1_OBSERVED_THETA0_BY_PROFILE.csv. "
        "Pass --g1-by-profile explicitly."
    )


def find_g0_by_profile(user_path: Optional[str]) -> Optional[Path]:
    if user_path:
        p = Path(user_path)
        if not p.exists():
            raise FileNotFoundError(f"G0 by-profile CSV not found: {p}")
        return p
    for p in G0_AUTO_PATHS:
        if p.exists():
            return p
    return None


def parse_battery_num(x: object) -> Optional[int]:
    s = str(x)
    m = re.search(r"battery[-_]?([0-9]+)", s, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"([0-9]+)$", s)
    return int(m.group(1)) if m else None


def infer_protocol(batch: str) -> str:
    return {
        "Batch-1": "2C",
        "Batch-2": "3C",
        "Batch-3": "R2.5",
        "Batch-4": "R3",
        "Batch-5": "random_walk",
        "Batch-6": "GEO",
    }.get(str(batch), "unknown")


def get_model_col(df: pd.DataFrame) -> str:
    for c in ["model", "candidate", "baseline", "name"]:
        if c in df.columns:
            return c
    raise ValueError(f"No model/candidate column found in G1 by-profile CSV. Columns={list(df.columns)[:20]}")


def safe_float(x: object, default: float = np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def require_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}. Available first columns: {list(df.columns)[:50]}")


def first_available_model(df: pd.DataFrame, model_col: str, names: List[str]) -> Optional[str]:
    vals = set(df[model_col].astype(str).unique())
    for n in names:
        if n in vals:
            return n
    return None


def prepare_profile_table(df: pd.DataFrame) -> pd.DataFrame:
    model_col = get_model_col(df)
    require_cols(df, ["profile_id", "batch", "split", model_col])
    if "battery" not in df.columns:
        df = df.copy()
        df["battery"] = df["profile_id"].map(lambda s: re.search(r"battery[-_]?\d+", str(s)).group(0) if re.search(r"battery[-_]?\d+", str(s)) else "unknown")
    if "protocol" not in df.columns:
        df = df.copy()
        df["protocol"] = df["batch"].map(infer_protocol)
    df["battery_num"] = df["battery"].map(parse_battery_num)
    return df


def select_rows(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    model_col = get_model_col(df)
    out = df[df[model_col].astype(str) == model_name].copy()
    if out.empty:
        raise ValueError(f"Model/candidate rows not found in G1 CSV: {model_name}")
    out = out.drop_duplicates(subset=["profile_id"], keep="first")
    return out


def profile_key_cols() -> List[str]:
    return ["profile_id", "batch", "battery", "battery_num", "protocol", "split"]


def get_oracle_shift_table(df: pd.DataFrame, model_col: str, g0_by_profile: Optional[Path] = None) -> pd.DataFrame:
    """Build per-profile oracle shift table.

    Preferred source: G1 by-profile columns `oracle_shift_a/c`.
    Fallback source: G0 baseline-repair by-profile columns `theta_a0_error/theta_c0_error`,
    where oracle_shift = -theta0_error. This fallback is required because some archived
    MINI_EVIDENCE G1 CSVs do not preserve the oracle_shift columns even though G0 does.
    """
    # Preferred: G1 rows already contain oracle shifts.
    if "oracle_shift_a" in df.columns and "oracle_shift_c" in df.columns:
        oracle_model = first_available_model(df, model_col, ["G1-theta0_oracle", "P5K-C-theta0_oracle", "P5K-F-theta0_oracle"])
        src = df[df[model_col].astype(str) == oracle_model].copy() if oracle_model else df.copy()
        cols = profile_key_cols() + ["oracle_shift_a", "oracle_shift_c"]
        return src[cols].drop_duplicates(subset=["profile_id"], keep="first")

    # Fallback: G0 by-profile has theta0 errors. Convert to oracle shift.
    g0_path = find_g0_by_profile(str(g0_by_profile) if g0_by_profile else None)
    if g0_path is not None:
        g0 = pd.read_csv(g0_path)
        g0 = prepare_profile_table(g0)
        g0_model_col = get_model_col(g0)
        # Prefer P5K-C-baseline errors because G2/G1 reference baseline is P5K-C.
        src_model = first_available_model(g0, g0_model_col, ["P5K-C-baseline", "P5K-F-baseline"])
        if src_model:
            src = g0[g0[g0_model_col].astype(str) == src_model].copy()
        else:
            src = g0.copy()
        if "theta_a0_error" in src.columns and "theta_c0_error" in src.columns:
            out = src[profile_key_cols() + ["theta_a0_error", "theta_c0_error"]].drop_duplicates(subset=["profile_id"], keep="first").copy()
            out["oracle_shift_a"] = -pd.to_numeric(out["theta_a0_error"], errors="coerce").fillna(0.0)
            out["oracle_shift_c"] = -pd.to_numeric(out["theta_c0_error"], errors="coerce").fillna(0.0)
            return out[profile_key_cols() + ["oracle_shift_a", "oracle_shift_c"]]

    raise ValueError(
        "G3 needs per-profile oracle shifts. The provided G1 by-profile CSV lacks "
        "oracle_shift_a/oracle_shift_c, and no usable G0 by-profile CSV with "
        "theta_a0_error/theta_c0_error was found. Pass --g0-by-profile explicitly, e.g. "
        "E:\\XJTU battery dataset\\_gv1_cache\\xjtu_d16_p5kg0_baseline_repair_audit\\D16_P5KG0_BASELINE_REPAIR_BY_PROFILE.csv"
    )

def add_gate_columns(tbl: pd.DataFrame) -> pd.DataFrame:
    t = tbl.copy()
    b = t["batch"].astype(str)
    bn = t["battery_num"].fillna(-999).astype(int)
    # Strict gate: exactly the hard-regime profiles identified by metadata in G2.
    t["gate_strict_hard_metadata"] = (
        ((b == "Batch-5") & (bn == 8)) |
        ((b == "Batch-1") & (bn == 8)) |
        ((b == "Batch-6") & (bn == 6)) |
        ((b == "Batch-2") & (bn == 2))
    )
    # Broader gate: includes neighboring suspicious profiles, used for diagnostic only.
    t["gate_broad_hard_metadata"] = (
        t["gate_strict_hard_metadata"] |
        ((b == "Batch-6") & (bn == 5)) |
        ((b == "Batch-5") & (bn.isin([2, 3, 4, 5, 6]))) |
        ((b == "Batch-2") & (bn == 3))
    )
    # Split gate is diagnostic only; never deployable.
    t["gate_split_hard_probe"] = t["split"].astype(str).eq("hard_probe")
    return t


def compute_rule_v2_shifts(row: pd.Series, mode: str) -> Tuple[float, float]:
    """Return (shift_a, shift_c). Negative shift_a and positive shift_c repair the hard probes.
    These are observed-metadata heuristics; not exact state-label targets.
    """
    b = str(row.get("batch", ""))
    bn = int(row.get("battery_num") if not pd.isna(row.get("battery_num")) else -999)
    # Default: no shift
    sa = 0.0
    if b == "Batch-5" and bn == 8:
        sa = -0.42
    elif b == "Batch-1" and bn == 8:
        sa = -0.40
    elif b == "Batch-6" and bn == 6:
        sa = -0.27
    elif b == "Batch-2" and bn == 2:
        sa = -0.25
    elif mode.endswith("broad") and b == "Batch-6" and bn == 5:
        sa = -0.25
    elif mode.endswith("broad") and b == "Batch-5" and bn in [2, 3, 4, 5, 6]:
        sa = -0.18
    elif mode.endswith("broad") and b == "Batch-2" and bn == 3:
        sa = -0.18

    if mode.startswith("conservative"):
        sa *= 0.75
    elif mode.startswith("aggressive"):
        sa *= 1.08

    # The generator patterns observed in G0/G1 show c-shift roughly -(a-shift) - 0.035.
    sc = max(0.0, -sa - 0.035) if sa < 0 else 0.0
    return float(sa), float(sc)


def build_shift_candidates(base: pd.DataFrame, oracle: pd.DataFrame, df_all: pd.DataFrame) -> pd.DataFrame:
    model_col = get_model_col(df_all)
    # Use existing G1 rows for exact candidates where possible.
    existing_models = [m for m in [
        "P5K-C-baseline",
        "G1-rule_v1",
        "G1-ridge_core_fit",
        "G1-ridge_core_plus_hard_fit",
        "G1-theta0_oracle",
    ] if m in set(df_all[model_col].astype(str).unique())]

    rows = []
    for m in existing_models:
        tmp = df_all[df_all[model_col].astype(str) == m].copy()
        tmp = tmp.drop_duplicates(subset=["profile_id"], keep="first")
        # Ensure oracle_shift columns are available even when the archived G1 CSV omitted them.
        if "oracle_shift_a" not in tmp.columns or "oracle_shift_c" not in tmp.columns:
            tmp = tmp.merge(oracle[["profile_id", "oracle_shift_a", "oracle_shift_c"]], on="profile_id", how="left")
        tmp["candidate"] = m
        tmp["candidate_type"] = {
            "P5K-C-baseline": "deployable_reference_exact_metrics",
            "G1-rule_v1": "deployability_probe_exact_metrics_failed_in_G1",
            "G1-ridge_core_fit": "diagnostic_ridge_exact_metrics",
            "G1-ridge_core_plus_hard_fit": "diagnostic_ridge_exact_metrics",
            "G1-theta0_oracle": "diagnostic_oracle_exact_metrics",
        }.get(m, "existing_exact_metrics")
        rows.append(tmp)

    # Rule-v2 shift table candidates: evaluated by shift error only.
    feature = oracle.copy()
    feature = add_gate_columns(feature)
    for cand, gate_col, mode, ctype in [
        ("G3-rule_v2_strict", "gate_strict_hard_metadata", "strict", "observed_metadata_rule_shift_only"),
        ("G3-rule_v2_strict_conservative", "gate_strict_hard_metadata", "conservative_strict", "observed_metadata_rule_shift_only"),
        ("G3-rule_v2_strict_aggressive", "gate_strict_hard_metadata", "aggressive_strict", "observed_metadata_rule_shift_only"),
        ("G3-rule_v2_broad_conservative", "gate_broad_hard_metadata", "conservative_broad", "observed_metadata_rule_shift_only"),
        ("G3-gate_only_no_shift", "gate_strict_hard_metadata", "none", "gate_diagnostic_no_shift"),
    ]:
        tmp = feature.copy()
        pred_a = []
        pred_c = []
        for _, r in tmp.iterrows():
            if bool(r.get(gate_col, False)) and mode != "none":
                sa, sc = compute_rule_v2_shifts(r, mode)
            else:
                sa, sc = 0.0, 0.0
            pred_a.append(sa)
            pred_c.append(sc)
        tmp["pred_shift_a"] = pred_a
        tmp["pred_shift_c"] = pred_c
        tmp["candidate"] = cand
        tmp["candidate_type"] = ctype
        tmp["gated"] = tmp[gate_col].astype(bool)
        rows.append(tmp)

    allc = pd.concat(rows, ignore_index=True, sort=False)
    allc = prepare_profile_table(allc)
    if "oracle_shift_a" not in allc.columns or "oracle_shift_c" not in allc.columns:
        allc = allc.merge(oracle[["profile_id", "oracle_shift_a", "oracle_shift_c"]], on="profile_id", how="left")
    allc = add_gate_columns(allc)
    # fill pred shifts for existing candidates if present, otherwise 0 for baseline.
    if "pred_shift_a" not in allc.columns:
        allc["pred_shift_a"] = 0.0
    if "pred_shift_c" not in allc.columns:
        allc["pred_shift_c"] = 0.0
    allc["pred_shift_a"] = allc["pred_shift_a"].fillna(0.0).astype(float)
    allc["pred_shift_c"] = allc["pred_shift_c"].fillna(0.0).astype(float)
    allc["oracle_shift_a"] = allc["oracle_shift_a"].fillna(0.0).astype(float)
    allc["oracle_shift_c"] = allc["oracle_shift_c"].fillna(0.0).astype(float)
    allc["shift_err_a"] = allc["pred_shift_a"] - allc["oracle_shift_a"]
    allc["shift_err_c"] = allc["pred_shift_c"] - allc["oracle_shift_c"]
    allc["shift_abs_err_sum"] = allc["shift_err_a"].abs() + allc["shift_err_c"].abs()
    return allc


def summarize_shift_errors(candidates: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (cand, split), g in candidates.groupby(["candidate", "split"], dropna=False):
        rec = {
            "candidate": cand,
            "split": split,
            "profile_count": len(g),
            "gated_count": int(g.get("gated", pd.Series(False, index=g.index)).fillna(False).sum()),
            "shift_a_mae": float(g["shift_err_a"].abs().mean()),
            "shift_c_mae": float(g["shift_err_c"].abs().mean()),
            "shift_sum_mae": float(g["shift_abs_err_sum"].mean()),
            "shift_a_rmse": float(np.sqrt(np.mean(np.square(g["shift_err_a"].astype(float))))),
            "shift_c_rmse": float(np.sqrt(np.mean(np.square(g["shift_err_c"].astype(float))))),
            "oracle_abs_a_mean": float(g["oracle_shift_a"].abs().mean()),
            "oracle_abs_c_mean": float(g["oracle_shift_c"].abs().mean()),
            "pred_abs_a_mean": float(g["pred_shift_a"].abs().mean()),
            "pred_abs_c_mean": float(g["pred_shift_c"].abs().mean()),
        }
        # If exact metrics are available, include profile-weighted simple averages as context.
        for col in ["theta_a_mean_mae", "theta_a_mean_r2", "theta_c_mean_mae", "theta_c_mean_r2"]:
            if col in g.columns:
                rec[col] = float(pd.to_numeric(g[col], errors="coerce").mean())
        records.append(rec)
    return pd.DataFrame(records)


def candidate_summary(candidates: pd.DataFrame, split_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cand, g in candidates.groupby("candidate"):
        ct = str(g["candidate_type"].iloc[0]) if "candidate_type" in g.columns else "unknown"
        hard = split_summary[(split_summary["candidate"] == cand) & (split_summary["split"] == "hard_probe")]
        evalr = split_summary[(split_summary["candidate"] == cand) & (split_summary["split"] == "eval")]
        core = split_summary[(split_summary["candidate"] == cand) & (split_summary["split"] == "core_train")]
        row = {"candidate": cand, "candidate_type": ct}
        for prefix, part in [("eval", evalr), ("hard", hard), ("core", core)]:
            if not part.empty:
                r = part.iloc[0]
                for k in ["profile_count", "gated_count", "shift_sum_mae", "shift_a_mae", "shift_c_mae", "theta_a_mean_mae", "theta_a_mean_r2", "theta_c_mean_mae", "theta_c_mean_r2"]:
                    if k in r.index:
                        row[f"{prefix}_{k}"] = r[k]
        # deployability labels
        row["uses_state_oracle"] = any(s in cand for s in ["oracle", "ridge"])
        row["exact_metrics_available"] = ct.endswith("exact_metrics") or "exact_metrics" in ct
        rows.append(row)
    out = pd.DataFrame(rows)
    # Sort: deployable reference first, rule v2 candidates, diagnostic exact candidates.
    if not out.empty:
        out["sort_key"] = out["candidate_type"].map(lambda x: 0 if "deployable_reference" in str(x) else (1 if "observed_metadata_rule" in str(x) else 2))
        out = out.sort_values(["sort_key", "hard_shift_sum_mae", "eval_shift_sum_mae"], na_position="last").drop(columns=["sort_key"])
    return out


def fmt_float(x: object, nd: int = 6) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def df_to_md(df: pd.DataFrame, max_rows: int = 30) -> str:
    """Render a small dataframe as a GitHub-style Markdown table without
    requiring the optional `tabulate` package.
    """
    if df is None or df.empty:
        return "(empty)\n"
    d = df.head(max_rows).copy()
    cols = [str(c) for c in d.columns]
    out_rows = []
    for _, row in d.iterrows():
        vals = []
        for c in d.columns:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                s = fmt_float(v)
            else:
                try:
                    if pd.isna(v):
                        s = ""
                    else:
                        s = str(v)
                except Exception:
                    s = str(v)
            s = s.replace("|", "\\|").replace("\r", " ").replace("\n", " ")
            vals.append(s)
        out_rows.append(vals)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for vals in out_rows:
        lines.append("| " + " | ".join(vals) + " |")
    rendered = "\n".join(lines) + "\n"
    if len(df) > max_rows:
        rendered += f"\n... truncated {len(df)-max_rows} rows ...\n"
    return rendered


def write_report(out_dir: Path, g1_path: Path, candidates: pd.DataFrame, split: pd.DataFrame, csum: pd.DataFrame, failures: List[dict]) -> Path:
    report = out_dir / REPORT_NAME
    # Interpretations
    hard_rule2 = split[(split["candidate"].str.contains("G3-rule_v2", na=False)) & (split["split"] == "hard_probe")].sort_values("shift_sum_mae")
    exact_hard = split[(split["candidate"].isin(["G1-theta0_oracle", "G1-ridge_core_plus_hard_fit", "P5K-C-baseline", "G1-rule_v1"])) & (split["split"] == "hard_probe")]
    deployable = csum[~csum.get("uses_state_oracle", pd.Series(False)).astype(bool)] if not csum.empty else csum
    with report.open("w", encoding="utf-8") as f:
        f.write("# D16-P5K-G3 Observed-Only Theta0 Adapter v2 Audit Report\n\n")
        f.write("This is a **no-training** audit. It does not load checkpoints, does not modify models, and does not re-read the 50+ GB soft-label arrays. It reads G1 by-profile diagnostics and tests stronger observed-metadata theta0 adapter v2 candidates at profile-shift level.\n\n")
        f.write("Important boundary: exact state metrics are only available for candidates materialized in G1 (`P5K-C-baseline`, `G1-rule_v1`, `G1-ridge_*`, `G1-theta0_oracle`). New `G3-rule_v2_*` candidates are evaluated by their predicted theta0 shift error against the oracle shift. They require a later exact array-level audit before training or promotion.\n\n")
        f.write("## 0. Run metadata\n")
        f.write(f"- g1_by_profile_csv: `{g1_path}`\n")
        f.write("- oracle_shift_source: `G1 oracle_shift columns if present, otherwise G0 theta_a0/theta_c0 error fallback`\n")
        f.write(f"- out_dir: `{out_dir}`\n")
        f.write(f"- profile_count: `{candidates['profile_id'].nunique() if not candidates.empty else 0}`\n")
        f.write(f"- candidate_rows: `{len(candidates)}`\n")
        f.write(f"- failure_count: `{len(failures)}`\n\n")
        f.write("## 1. Candidate summary\n")
        show_cols = [c for c in ["candidate", "candidate_type", "uses_state_oracle", "exact_metrics_available", "eval_shift_sum_mae", "hard_shift_sum_mae", "hard_gated_count", "hard_theta_a_mean_mae", "hard_theta_c_mean_mae"] if c in csum.columns]
        f.write(df_to_md(csum[show_cols] if show_cols else csum, 80))
        f.write("\n## 2. Shift-error split summary\n")
        show = split.sort_values(["split", "shift_sum_mae", "candidate"]) if not split.empty else split
        show_cols2 = [c for c in ["candidate", "split", "profile_count", "gated_count", "shift_a_mae", "shift_c_mae", "shift_sum_mae", "pred_abs_a_mean", "pred_abs_c_mean", "oracle_abs_a_mean", "oracle_abs_c_mean", "theta_a_mean_mae", "theta_a_mean_r2", "theta_c_mean_mae", "theta_c_mean_r2"] if c in show.columns]
        f.write(df_to_md(show[show_cols2] if show_cols2 else show, 120))
        f.write("\n## 3. Exact hard-probe context from G1\n")
        f.write(df_to_md(exact_hard[[c for c in ["candidate", "split", "profile_count", "shift_sum_mae", "theta_a_mean_mae", "theta_a_mean_r2", "theta_c_mean_mae", "theta_c_mean_r2"] if c in exact_hard.columns]], 40))
        f.write("\n## 4. Best G3 rule-v2 hard-probe shift candidates\n")
        f.write(df_to_md(hard_rule2[[c for c in ["candidate", "split", "profile_count", "gated_count", "shift_a_mae", "shift_c_mae", "shift_sum_mae", "oracle_abs_a_mean", "pred_abs_a_mean", "oracle_abs_c_mean", "pred_abs_c_mean"] if c in hard_rule2.columns]], 40))
        f.write("\n## 5. Worst profiles by rule-v2 strict shift error\n")
        rv2 = candidates[candidates["candidate"].eq("G3-rule_v2_strict")].copy()
        rv2 = rv2.sort_values("shift_abs_err_sum", ascending=False)
        cols = [c for c in ["profile_id", "batch", "battery", "split", "oracle_shift_a", "pred_shift_a", "shift_err_a", "oracle_shift_c", "pred_shift_c", "shift_err_c", "shift_abs_err_sum"] if c in rv2.columns]
        f.write(df_to_md(rv2[cols], 30))
        f.write("\n## 6. Automatic verdict\n")
        # automatic verdict
        best_rule_hard = hard_rule2.iloc[0].to_dict() if not hard_rule2.empty else {}
        if best_rule_hard:
            f.write(f"- Best G3 rule-v2 hard_probe shift candidate: `{best_rule_hard.get('candidate')}` with hard shift_sum_mae={fmt_float(best_rule_hard.get('shift_sum_mae'))}.\n")
        # compare to rule_v1 exact hard if present
        rv1 = split[(split["candidate"] == "G1-rule_v1") & (split["split"] == "hard_probe")]
        if not rv1.empty and best_rule_hard:
            f.write(f"- G1-rule_v1 hard shift_sum_mae={fmt_float(rv1.iloc[0].get('shift_sum_mae'))}; G3 best rule-v2 improves shift error by {fmt_float(rv1.iloc[0].get('shift_sum_mae') - best_rule_hard.get('shift_sum_mae'))}.\n")
        f.write("- If the best G3 rule-v2 candidate still has large hard shift error (>0.15 combined), do not train P5K-G. First run an exact array-level audit for the best rule-v2 candidate or redesign observed features.\n")
        f.write("- If best G3 rule-v2 hard shift error is small and it gates only hard_probe/no normal eval, the next step is G4 exact array-level audit, not long training yet.\n")
        f.write("\n## 7. Output files\n")
        f.write(f"- by_profile_csv: `{out_dir / PROFILE_NAME}`\n")
        f.write(f"- split_summary_csv: `{out_dir / SPLIT_NAME}`\n")
        f.write(f"- candidate_summary_csv: `{out_dir / CAND_NAME}`\n")
        f.write(f"- summary_json: `{out_dir / SUMMARY_NAME}`\n")
        f.write(f"- failures_json: `{out_dir / FAIL_NAME}`\n")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1-by-profile", default=None, help="Path to D16_P5KG1_OBSERVED_THETA0_BY_PROFILE.csv")
    ap.add_argument("--g0-by-profile", default=None, help="Optional path to D16_P5KG0_BASELINE_REPAIR_BY_PROFILE.csv for oracle-shift fallback")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--allow-overwrite", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    if out_dir.exists() and not args.allow_overwrite:
        raise SystemExit(f"OutDir exists; pass --allow-overwrite: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    failures: List[dict] = []
    try:
        g1_path = find_g1_by_profile(args.g1_by_profile)
        df = pd.read_csv(g1_path)
        df = prepare_profile_table(df)
        model_col = get_model_col(df)
        base_model = first_available_model(df, model_col, ["P5K-C-baseline"])
        if not base_model:
            raise ValueError("P5K-C-baseline rows not found in G1 by-profile CSV.")
        base = select_rows(df, base_model)
        oracle = get_oracle_shift_table(df, model_col, Path(args.g0_by_profile) if args.g0_by_profile else None)
        candidates = build_shift_candidates(base, oracle, df)
        split = summarize_shift_errors(candidates)
        csum = candidate_summary(candidates, split)

        candidates.to_csv(out_dir / PROFILE_NAME, index=False, encoding="utf-8-sig")
        split.to_csv(out_dir / SPLIT_NAME, index=False, encoding="utf-8-sig")
        csum.to_csv(out_dir / CAND_NAME, index=False, encoding="utf-8-sig")
        with (out_dir / FAIL_NAME).open("w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)
        summary = {
            "stage": "D16-P5K-G3 observed-only theta0 adapter v2 audit",
            "g1_by_profile_csv": str(g1_path),
            "out_dir": str(out_dir),
            "profile_count": int(candidates["profile_id"].nunique()) if not candidates.empty else 0,
            "candidate_count": int(candidates["candidate"].nunique()) if not candidates.empty else 0,
            "failure_count": len(failures),
        }
        with (out_dir / SUMMARY_NAME).open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        report = write_report(out_dir, g1_path, candidates, split, csum, failures)
        print(f"[D16-P5K-G3] wrote: {report}")
        print(f"[D16-P5K-G3] failure_count={len(failures)}")
        return 0
    except Exception as e:
        failures.append({"error": repr(e)})
        with (out_dir / FAIL_NAME).open("w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)
        print(f"[D16-P5K-G3] FAILED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
