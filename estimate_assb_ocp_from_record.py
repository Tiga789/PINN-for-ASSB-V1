#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据全电池低倍率恒流充放电数据，结合 ASSB 先验，估计两电极 OCP。

改进点（相对前一版）：
1) 不再简单平均所有低倍率段，而是先筛选“完整且代表性”的低倍率循环。
2) 用低倍率充/放电后的短静置末端电压，对全电池 pOCV 端点做锚定。
3) 与 10 s 原始数据对齐时，不再把最前面的未知初始搁置段强行回填 SOC。
4) 示例图改为展示“被选中的低倍率循环窗口”，避免误导性地画整段首部原始记录。

说明：
- 正极：按 NMC811 复合正极处理，但 OCP 主体仍视作 NMC811 本征 OCP 的工作窗口。
- 负极：第一版仍按 Li-In 主平台近似处理，属于工程先验，不是完整单电极实测 OCP。
- 本脚本独立运行，不依赖 TensorFlow / PyTorch。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 你本地的默认路径
# =========================
DEFAULT_CSV = r"C:\Users\Tiga_QJW\Desktop\ZHB_realDATA\record_extracted.csv"
DEFAULT_OUTDIR = r"./ocp_estimation_outputs"

# =========================
# 已知先验（可按需微调）
# =========================
POSITIVE_TOP_OCP_V = 4.30         # NMC811 充满附近的正极上限电位先验（vs. Li/Li+）
NEGATIVE_LIIN_PLATEAU_V = 0.62    # Li-In 主平台电位先验（vs. Li/Li+）
POSITIVE_MIN_REASONABLE_V = 2.60  # 仅用于提示，不强制
POSITIVE_MAX_REASONABLE_V = 4.35  # 仅用于提示，不强制

SMOOTH_WINDOW = 21
GRID_N = 501
MIN_POINTS_PER_STEP = 50

# 中文列名
COL_RECORD = "数据序号"
COL_CYCLE = "循环号"
COL_STEP = "工步号"
COL_STEP_TYPE = "工步类型"
COL_TIME = "时间"
COL_TOTAL_TIME = "总时间"
COL_CURRENT = "电流(A)"
COL_VOLTAGE = "电压(V)"
COL_CAP = "容量(Ah)"
COL_CHG_CAP = "充电容量(Ah)"
COL_DCH_CAP = "放电容量(Ah)"
COL_ABS_TIME = "绝对时间"
COL_POWER = "功率(W)"

CHARGE_LABEL = "恒流充电"
DISCHARGE_LABEL = "恒流放电"
REST_LABEL = "搁置"
ACTIVE_LABELS = [CHARGE_LABEL, DISCHARGE_LABEL]


# -------------------------
# 基础工具
# -------------------------
def moving_average(y: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if window <= 1 or len(y) < window:
        return y.copy()
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(ypad, kernel, mode="valid")


def ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列：{missing}")


def parse_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    ensure_columns(
        df,
        [
            COL_CYCLE,
            COL_STEP,
            COL_STEP_TYPE,
            COL_CURRENT,
            COL_VOLTAGE,
            COL_CHG_CAP,
            COL_DCH_CAP,
        ],
    )

    numeric_cols = [
        COL_CYCLE,
        COL_STEP,
        COL_CURRENT,
        COL_VOLTAGE,
        COL_CHG_CAP,
        COL_DCH_CAP,
    ]
    for c in [COL_CAP, COL_POWER, COL_RECORD]:
        if c in df.columns:
            numeric_cols.append(c)
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if COL_ABS_TIME in df.columns:
        df[COL_ABS_TIME] = pd.to_datetime(df[COL_ABS_TIME], errors="coerce")

    df = df.dropna(subset=[COL_CYCLE, COL_STEP, COL_STEP_TYPE, COL_CURRENT, COL_VOLTAGE]).copy()
    df[COL_STEP_TYPE] = df[COL_STEP_TYPE].astype(str)
    df[COL_CYCLE] = df[COL_CYCLE].astype(int)
    df[COL_STEP] = df[COL_STEP].astype(int)

    if COL_RECORD not in df.columns:
        df[COL_RECORD] = np.arange(len(df), dtype=int) + 1

    return df


def detect_sample_interval_seconds(df: pd.DataFrame) -> float:
    if COL_ABS_TIME in df.columns and df[COL_ABS_TIME].notna().sum() > 2:
        diffs = df[COL_ABS_TIME].sort_values().diff().dt.total_seconds().dropna()
        diffs = diffs[(diffs > 0) & (diffs < 3600)]
        if len(diffs) > 0:
            return float(diffs.mode().iloc[0])
    return 10.0


def detect_low_rate_current(df: pd.DataFrame) -> float:
    active = df[df[COL_STEP_TYPE].isin(ACTIVE_LABELS)].copy()
    mags = np.round(np.abs(active[COL_CURRENT].values.astype(float)), 9)
    mags = mags[mags > 0]
    if len(mags) == 0:
        raise ValueError("没有检测到非零充放电电流。")
    vc = pd.Series(mags).value_counts().sort_index()
    vc = vc[vc >= MIN_POINTS_PER_STEP]
    if len(vc) == 0:
        # 回退：如果所有电流点都太少，就取最小非零电流
        return float(np.min(mags))
    return float(vc.index[0])


def build_step_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby([COL_CYCLE, COL_STEP, COL_STEP_TYPE], as_index=False)
    out = g.agg(
        n=(COL_VOLTAGE, "size"),
        current_mean=(COL_CURRENT, "mean"),
        v_start=(COL_VOLTAGE, "first"),
        v_end=(COL_VOLTAGE, "last"),
        v_min=(COL_VOLTAGE, "min"),
        v_max=(COL_VOLTAGE, "max"),
        qchg_start=(COL_CHG_CAP, "first"),
        qchg_end=(COL_CHG_CAP, "last"),
        qdch_start=(COL_DCH_CAP, "first"),
        qdch_end=(COL_DCH_CAP, "last"),
        rec_start=(COL_RECORD, "min"),
        rec_end=(COL_RECORD, "max"),
    )
    out["qchg_step"] = out["qchg_end"] - out["qchg_start"]
    out["qdch_step"] = out["qdch_end"] - out["qdch_start"]
    out["i_abs"] = np.abs(out["current_mean"].astype(float))
    return out


# -------------------------
# 低倍率循环筛选与端点锚定
# -------------------------
def find_immediate_next_rest(df_cycle: pd.DataFrame, step_no: int) -> Optional[pd.DataFrame]:
    rest_steps = sorted(df_cycle.loc[df_cycle[COL_STEP_TYPE] == REST_LABEL, COL_STEP].unique())
    later = [s for s in rest_steps if s > step_no]
    if not later:
        return None
    return df_cycle[df_cycle[COL_STEP] == later[0]].copy()


def select_representative_low_rate_cycles(
    df: pd.DataFrame,
    steps: pd.DataFrame,
    low_current: float,
    tol: float = 0.2,
) -> Tuple[List[int], pd.DataFrame, Dict[str, float]]:
    low_steps = steps[
        (steps[COL_STEP_TYPE].isin(ACTIVE_LABELS))
        & (np.abs(steps["i_abs"] - low_current) <= max(low_current * tol, 1e-12))
        & (steps["n"] >= MIN_POINTS_PER_STEP)
    ].copy()

    candidate_rows = []
    for cyc in sorted(low_steps[COL_CYCLE].unique()):
        c_active = low_steps[low_steps[COL_CYCLE] == cyc].copy()
        ch = c_active[c_active[COL_STEP_TYPE] == CHARGE_LABEL].sort_values(COL_STEP)
        dch = c_active[c_active[COL_STEP_TYPE] == DISCHARGE_LABEL].sort_values(COL_STEP)
        if len(ch) == 0 or len(dch) == 0:
            continue
        ch_row = ch.iloc[0]
        dch_row = dch.iloc[0]
        candidate_rows.append(
            {
                COL_CYCLE: int(cyc),
                "charge_step": int(ch_row[COL_STEP]),
                "discharge_step": int(dch_row[COL_STEP]),
                "charge_n": int(ch_row["n"]),
                "discharge_n": int(dch_row["n"]),
                "charge_start_v": float(ch_row["v_start"]),
                "charge_end_v": float(ch_row["v_end"]),
                "discharge_start_v": float(dch_row["v_start"]),
                "discharge_end_v": float(dch_row["v_end"]),
                "qchg_step": float(ch_row["qchg_step"]),
                "qdch_step": float(dch_row["qdch_step"]),
            }
        )

    candidate = pd.DataFrame(candidate_rows)
    if len(candidate) == 0:
        raise ValueError("没有找到完整的低倍率充放电循环。")

    med_qchg = float(candidate["qchg_step"].median())
    med_qdch = float(candidate["qdch_step"].median())
    med_ch_start = float(candidate["charge_start_v"].median())
    med_dch_end = float(candidate["discharge_end_v"].median())
    med_ch_end = float(candidate["charge_end_v"].median())
    med_dch_start = float(candidate["discharge_start_v"].median())

    good = candidate[
        candidate["qchg_step"].between(0.85 * med_qchg, 1.15 * med_qchg)
        & candidate["qdch_step"].between(0.85 * med_qdch, 1.15 * med_qdch)
        & (candidate["charge_start_v"] >= med_ch_start - 0.15)
        & (np.abs(candidate["charge_end_v"] - med_ch_end) <= 0.05)
        & (np.abs(candidate["discharge_end_v"] - med_dch_end) <= 0.05)
        & (np.abs(candidate["discharge_start_v"] - med_dch_start) <= 0.08)
    ].copy()

    if len(good) < 2:
        # 回退：只排除明显异常的起始电压 outlier
        good = candidate[candidate["charge_start_v"] >= med_ch_start - 0.25].copy()
    if len(good) < 1:
        good = candidate.copy()

    selected_cycles = [int(x) for x in good[COL_CYCLE].tolist()]

    # 收集静置端点
    top_rest_ends = []
    bottom_rest_ends = []
    for cyc in selected_cycles:
        cdf = df[df[COL_CYCLE] == cyc].copy()
        row = good[good[COL_CYCLE] == cyc].iloc[0]
        ch_rest = find_immediate_next_rest(cdf, int(row["charge_step"]))
        dch_rest = find_immediate_next_rest(cdf, int(row["discharge_step"]))
        if ch_rest is not None and len(ch_rest) > 0:
            top_rest_ends.append(float(ch_rest[COL_VOLTAGE].iloc[-1]))
        if dch_rest is not None and len(dch_rest) > 0:
            bottom_rest_ends.append(float(dch_rest[COL_VOLTAGE].iloc[-1]))

    anchors = {
        "top_rest_end_v": float(np.median(top_rest_ends)) if top_rest_ends else float(med_ch_end),
        "bottom_rest_end_v": float(np.median(bottom_rest_ends)) if bottom_rest_ends else float(med_dch_end),
        "median_charge_capacity_Ah": med_qchg,
        "median_discharge_capacity_Ah": med_qdch,
    }
    return selected_cycles, good, anchors


# -------------------------
# 曲线构建
# -------------------------
def build_soc_curve_for_step(step_df: pd.DataFrame, step_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """返回 z(0~1, 充满态=1) 与对应电压 V。"""
    step_df = step_df.copy().reset_index(drop=True)
    v = step_df[COL_VOLTAGE].astype(float).values

    if step_type == CHARGE_LABEL:
        q = step_df[COL_CHG_CAP].astype(float).values
        q = q - q[0]
        q_end = max(q[-1], 1e-15)
        z = q / q_end
    elif step_type == DISCHARGE_LABEL:
        q = step_df[COL_DCH_CAP].astype(float).values
        q = q - q[0]
        q_end = max(q[-1], 1e-15)
        z = 1.0 - q / q_end
    else:
        raise ValueError(f"不支持的工步类型：{step_type}")

    z = np.clip(z, 0.0, 1.0)
    return z, v


def average_branches_from_selected_cycles(
    df: pd.DataFrame,
    cycle_table: pd.DataFrame,
    grid_n: int = GRID_N,
) -> Dict[str, np.ndarray]:
    grid = np.linspace(0.0, 1.0, grid_n)
    charge_curves = []
    discharge_curves = []
    charge_meta = []
    discharge_meta = []

    for _, row in cycle_table.iterrows():
        cyc = int(row[COL_CYCLE])
        ch_step = int(row["charge_step"])
        dch_step = int(row["discharge_step"])

        sdf_ch = df[(df[COL_CYCLE] == cyc) & (df[COL_STEP] == ch_step) & (df[COL_STEP_TYPE] == CHARGE_LABEL)].copy()
        sdf_dch = df[(df[COL_CYCLE] == cyc) & (df[COL_STEP] == dch_step) & (df[COL_STEP_TYPE] == DISCHARGE_LABEL)].copy()

        if len(sdf_ch) >= MIN_POINTS_PER_STEP:
            z, v = build_soc_curve_for_step(sdf_ch, CHARGE_LABEL)
            order = np.argsort(z)
            z = z[order]
            v = v[order]
            keep = ~pd.Series(np.round(z, 12)).duplicated(keep="first")
            z = z[keep.values]
            v = v[keep.values]
            if len(z) >= 10:
                charge_curves.append(np.interp(grid, z, v))
                charge_meta.append((cyc, ch_step, float(sdf_ch[COL_CURRENT].mean()), len(sdf_ch)))

        if len(sdf_dch) >= MIN_POINTS_PER_STEP:
            z, v = build_soc_curve_for_step(sdf_dch, DISCHARGE_LABEL)
            order = np.argsort(z)
            z = z[order]
            v = v[order]
            keep = ~pd.Series(np.round(z, 12)).duplicated(keep="first")
            z = z[keep.values]
            v = v[keep.values]
            if len(z) >= 10:
                discharge_curves.append(np.interp(grid, z, v))
                discharge_meta.append((cyc, dch_step, float(sdf_dch[COL_CURRENT].mean()), len(sdf_dch)))

    if not charge_curves or not discharge_curves:
        raise ValueError("筛选后的低倍率循环不足，无法构造 pOCV。")

    vchg = np.median(np.vstack(charge_curves), axis=0)
    vdch = np.median(np.vstack(discharge_curves), axis=0)
    vavg = 0.5 * (vchg + vdch)

    vchg = moving_average(vchg)
    vdch = moving_average(vdch)
    vavg = moving_average(vavg)

    return {
        "z": grid,
        "v_charge_mean": vchg,
        "v_discharge_mean": vdch,
        "v_avg": vavg,
        "charge_meta": charge_meta,
        "discharge_meta": discharge_meta,
    }


def anchor_pocv_with_relaxed_endpoints(
    z: np.ndarray,
    vavg: np.ndarray,
    top_rest_end_v: float,
    bottom_rest_end_v: float,
) -> np.ndarray:
    v0 = float(vavg[0])
    v1 = float(vavg[-1])
    if abs(v1 - v0) < 1e-9:
        return np.full_like(vavg, 0.5 * (top_rest_end_v + bottom_rest_end_v))
    vp = bottom_rest_end_v + (vavg - v0) * (top_rest_end_v - bottom_rest_end_v) / (v1 - v0)
    vp = moving_average(vp)
    vp = np.maximum.accumulate(vp)  # 保证随 SOC 单调不降
    vp[0] = bottom_rest_end_v
    vp[-1] = top_rest_end_v
    return vp


def estimate_negative_ocp_single_plateau(
    v_charge_mean: np.ndarray,
    positive_top_ocp_v: float = POSITIVE_TOP_OCP_V,
) -> np.ndarray:
    """
    负极 Li-In 第一版：主平台近似。
    仍按 U_n ≈ U_p,top - V_charge,top 估计常数平台值。
    """
    v_top = float(np.nanmax(v_charge_mean))
    u_from_top = positive_top_ocp_v - v_top
    if 0.50 <= u_from_top <= 0.75:
        u0 = u_from_top
    else:
        u0 = NEGATIVE_LIIN_PLATEAU_V
    return np.full_like(v_charge_mean, u0, dtype=float)


def derive_positive_ocp(v_pocv: np.ndarray, u_n: np.ndarray) -> np.ndarray:
    u_p = v_pocv + u_n
    u_p = moving_average(u_p)
    u_p = np.maximum.accumulate(u_p)
    return u_p


# -------------------------
# 与原始时序对齐
# -------------------------
def build_cyclewise_soc_and_aligned_ocp(
    df: pd.DataFrame,
    z_grid: np.ndarray,
    u_p_grid: np.ndarray,
    u_n_grid: np.ndarray,
    v_pocv_grid: np.ndarray,
) -> pd.DataFrame:
    df2 = df.copy()
    df2["soc_cycle_est"] = np.nan

    for cyc, cdf in df2.groupby(COL_CYCLE):
        idx = cdf.index
        charge_mask = cdf[COL_STEP_TYPE] == CHARGE_LABEL
        discharge_mask = cdf[COL_STEP_TYPE] == DISCHARGE_LABEL

        qchg_total = None
        qdch_total = None
        if charge_mask.any():
            qchg = cdf.loc[charge_mask, COL_CHG_CAP].astype(float)
            qchg_total = float(qchg.max() - qchg.min())
        if discharge_mask.any():
            qdch = cdf.loc[discharge_mask, COL_DCH_CAP].astype(float)
            qdch_total = float(qdch.max() - qdch.min())

        qchg_total = qchg_total if (qchg_total is not None and qchg_total > 0) else None
        qdch_total = qdch_total if (qdch_total is not None and qdch_total > 0) else None

        soc = np.full(len(cdf), np.nan, dtype=float)
        stypes = cdf[COL_STEP_TYPE].astype(str).values
        qchg_vals = cdf[COL_CHG_CAP].astype(float).values
        qdch_vals = cdf[COL_DCH_CAP].astype(float).values
        qchg_min = float(cdf.loc[charge_mask, COL_CHG_CAP].min()) if charge_mask.any() else 0.0
        qdch_min = float(cdf.loc[discharge_mask, COL_DCH_CAP].min()) if discharge_mask.any() else 0.0

        for j, stype in enumerate(stypes):
            if stype == CHARGE_LABEL and qchg_total is not None:
                q = float(qchg_vals[j]) - qchg_min
                soc[j] = np.clip(q / qchg_total, 0.0, 1.0)
            elif stype == DISCHARGE_LABEL and qdch_total is not None:
                q = float(qdch_vals[j]) - qdch_min
                soc[j] = np.clip(1.0 - q / qdch_total, 0.0, 1.0)

        s = pd.Series(soc, index=idx, dtype=float)
        # 只在线段内部插值，不把最前面的未知搁置段强行回填
        s = s.interpolate(limit_area="inside")
        first_valid = s.first_valid_index()
        if first_valid is not None:
            s.loc[first_valid:] = s.loc[first_valid:].ffill()
        df2.loc[idx, "soc_cycle_est"] = s.values

    valid = df2["soc_cycle_est"].notna().values
    df2["U_p_ocp_est_V"] = np.nan
    df2["U_n_ocp_est_V"] = np.nan
    df2["V_pocv_est_V"] = np.nan
    df2.loc[valid, "U_p_ocp_est_V"] = np.interp(df2.loc[valid, "soc_cycle_est"].values, z_grid, u_p_grid)
    df2.loc[valid, "U_n_ocp_est_V"] = np.interp(df2.loc[valid, "soc_cycle_est"].values, z_grid, u_n_grid)
    df2.loc[valid, "V_pocv_est_V"] = np.interp(df2.loc[valid, "soc_cycle_est"].values, z_grid, v_pocv_grid)
    return df2


# -------------------------
# 输出与绘图
# -------------------------
def save_curve_csv(outdir: Path, z: np.ndarray, y: np.ndarray, name: str, y_name: str) -> None:
    pd.DataFrame({"soc_0to1": z, y_name: y}).to_csv(outdir / name, index=False, encoding="utf-8-sig")


def pick_example_cycle_window(df: pd.DataFrame, cycle_table: pd.DataFrame) -> pd.DataFrame:
    if len(cycle_table) == 0:
        return df.iloc[: min(5000, len(df))].copy()
    cyc = int(cycle_table.iloc[0][COL_CYCLE])
    ch_step = int(cycle_table.iloc[0]["charge_step"])
    dch_step = int(cycle_table.iloc[0]["discharge_step"])
    cdf = df[df[COL_CYCLE] == cyc].copy()
    end_step = dch_step
    next_rest = find_immediate_next_rest(cdf, dch_step)
    if next_rest is not None and len(next_rest) > 0:
        end_step = int(next_rest[COL_STEP].iloc[0])
    win = cdf[(cdf[COL_STEP] >= ch_step) & (cdf[COL_STEP] <= end_step)].copy()
    if len(win) == 0:
        win = cdf.copy()
    return win


def make_plots(
    outdir: Path,
    z: np.ndarray,
    vpocv: np.ndarray,
    up: np.ndarray,
    un: np.ndarray,
    vchg: np.ndarray,
    vdch: np.ndarray,
    example_df: pd.DataFrame,
) -> None:
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(z, vchg, label="Low-rate charge branch", alpha=0.70)
    plt.plot(z, vdch, label="Low-rate discharge branch", alpha=0.70)
    plt.plot(z, vpocv, label="Rest-anchored full-cell pOCV", linewidth=2.0)
    plt.plot(z, up, label="Estimated cathode OCP", linewidth=2.0)
    plt.plot(z, un, label="Estimated anode OCP", linewidth=2.0)
    plt.xlabel("soc_0to1")
    plt.ylabel("Voltage / V")
    plt.title("ASSB single-electrode OCP estimation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "ocp_curves.png", dpi=200)
    plt.close()

    sub = example_df.copy().reset_index(drop=True)
    x = np.arange(len(sub))
    plt.figure(figsize=(10, 5.2))
    plt.plot(x, sub[COL_VOLTAGE].astype(float).values, label="Raw full-cell voltage")
    if "V_pocv_est_V" in sub.columns:
        plt.plot(x, sub["V_pocv_est_V"].values, label="Estimated full-cell pOCV")
        plt.plot(x, sub["U_p_ocp_est_V"].values, label="Estimated cathode OCP")
        plt.plot(x, sub["U_n_ocp_est_V"].values, label="Estimated anode OCP")
    plt.xlabel("sample index within selected low-rate cycle")
    plt.ylabel("Voltage / V")
    plt.title("OCP estimates aligned with selected low-rate cycle")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "ocp_timeseries_example.png", dpi=200)
    plt.close()


# -------------------------
# 主流程
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="输入 CSV 路径")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="输出文件夹")
    parser.add_argument(
        "--positive-top-ocp",
        type=float,
        default=POSITIVE_TOP_OCP_V,
        help="正极上限 OCP 先验，默认 4.30 V vs. Li/Li+",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 文件：{csv_path}")

    print(f"[INFO] 读取 CSV: {csv_path}")
    df = parse_csv(str(csv_path))
    dt = detect_sample_interval_seconds(df)
    print(f"[INFO] 检测到采样间隔约为: {dt:.3f} s")

    low_current = detect_low_rate_current(df)
    print(f"[INFO] 检测到最低非零电流幅值: {low_current:.9f} A")

    steps = build_step_table(df)
    selected_cycles, selected_table, anchors = select_representative_low_rate_cycles(df, steps, low_current=low_current, tol=0.2)
    print(f"[INFO] 用于 pOCV 的代表性低倍率循环: {selected_cycles}")
    print(f"[INFO] 低倍率静置端点锚定: 下端 {anchors['bottom_rest_end_v']:.4f} V, 上端 {anchors['top_rest_end_v']:.4f} V")

    branches = average_branches_from_selected_cycles(df, selected_table, grid_n=GRID_N)
    z = branches["z"]
    vchg = branches["v_charge_mean"]
    vdch = branches["v_discharge_mean"]
    vpocv = anchor_pocv_with_relaxed_endpoints(z, branches["v_avg"], anchors["top_rest_end_v"], anchors["bottom_rest_end_v"])

    un = estimate_negative_ocp_single_plateau(v_charge_mean=vchg, positive_top_ocp_v=args.positive_top_ocp)
    up = derive_positive_ocp(vpocv, un)

    aligned = build_cyclewise_soc_and_aligned_ocp(df, z, up, un, vpocv)

    # 导出曲线 CSV
    pd.DataFrame({
        "soc_0to1": z,
        "V_charge_mean_V": vchg,
        "V_discharge_mean_V": vdch,
        "V_pocv_est_V": vpocv,
    }).to_csv(outdir / "fullcell_pocv_curve.csv", index=False, encoding="utf-8-sig")

    save_curve_csv(outdir, z, un, "negative_ocp_curve.csv", "U_n_ocp_est_V")
    save_curve_csv(outdir, z, up, "positive_ocp_curve.csv", "U_p_ocp_est_V")

    aligned.to_csv(outdir / "aligned_ocp_timeseries.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"x": z, "Ueq": un}).to_csv(outdir / "anEeq_est.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"x": z, "Ueq": up}).to_csv(outdir / "caEeq_est.csv", index=False, encoding="utf-8-sig")

    selected_table.to_csv(outdir / "lowrate_steps_used.csv", index=False, encoding="utf-8-sig")

    example_df = pick_example_cycle_window(aligned, selected_table)
    make_plots(outdir, z, vpocv, up, un, vchg, vdch, example_df)

    summary = {
        "input_csv": str(csv_path),
        "n_rows": int(len(df)),
        "sample_interval_s": float(dt),
        "detected_low_rate_current_A": float(low_current),
        "selected_low_rate_cycles": selected_cycles,
        "n_selected_cycles": int(len(selected_cycles)),
        "top_rest_end_v": float(anchors["top_rest_end_v"]),
        "bottom_rest_end_v": float(anchors["bottom_rest_end_v"]),
        "median_charge_capacity_Ah": float(anchors["median_charge_capacity_Ah"]),
        "median_discharge_capacity_Ah": float(anchors["median_discharge_capacity_Ah"]),
        "negative_ocp_model": "Li-In main plateau prior",
        "negative_ocp_constant_V": float(un[0]),
        "positive_ocp_min_V": float(np.min(up)),
        "positive_ocp_max_V": float(np.max(up)),
        "notes": [
            "这一版先筛选代表性低倍率循环，再用充/放电后的短静置末端电压对 pOCV 端点做锚定。",
            "负极仍是 Li-In 主平台近似，适合作为 SPM/PINN 的第一版先验；若后续要更细，可再升级为双/三平台模型。",
            "与 10 s 数据对齐时，不再把最前面的未知初始搁置段强行回填 SOC，因此示例图会更接近有效低倍率窗口。",
        ],
    }
    with open(outdir / "ocp_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[INFO] 输出完成。")
    print(f"[INFO] 输出目录: {outdir.resolve()}")
    print(f"[INFO] 负极主平台估计值: {un[0]:.4f} V")
    print(f"[INFO] 正极 OCP 范围: {np.min(up):.4f} ~ {np.max(up):.4f} V")

    if np.min(up) < POSITIVE_MIN_REASONABLE_V or np.max(up) > POSITIVE_MAX_REASONABLE_V:
        print("[WARN] 正极 OCP 超出常见范围，请检查：")
        print("       1) 低倍率循环是否识别正确")
        print("       2) Li-In 平台先验是否需要调整")
        print("       3) 该电池是否只使用了 NMC811 OCP 曲线的一部分工作窗口")


if __name__ == "__main__":
    main()
