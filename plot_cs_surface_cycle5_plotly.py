# -*- coding: utf-8 -*-
r"""
plot_cs_surface_cycle5_plotly.py

Purpose
-------
Generate interactive Plotly 3D surface HTML files for ASSB cycle5 ID101:
    1) cs_a true surface
    2) cs_a predicted surface
    3) cs_c true surface
    4) cs_c predicted surface

Default target project state
----------------------------
Project root should be:
    C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
Default evaluation directory:
    EvalFin_101_cycle5_v4_cbarAC_potentialBaseline
Default soft-label directory:
    Data/assb_soft_labels_cycle5_v4

The script is read-only. It does not train or overwrite model files.
It reads evaluation arrays / soft-label arrays; if prediction arrays are absent,
it can recompute predictions from ModelFin_101 and cache only the generated HTML/JSON outputs.

How to run on Windows, from project root
----------------------------------------
    D:\Anaconda\envs\torchgpu\python.exe .\plot_cs_surface_cycle5_plotly.py \
      --ocp_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs"

Notes
-----
- Plotly output is interactive in a web browser: drag to rotate, scroll/pinch to zoom.
- To avoid slow dragging, surfaces are downsampled for plotting only. Metrics are computed
  on the full-resolution arrays before downsampling.
- All Plotly text is requested as Times New Roman.
- Default truth colormap is Viridis, i.e. purple-to-yellow.
- Default prediction colormap is RdBu_r, i.e. blue-to-red.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------
# Default project-local settings
# -----------------------------
DEFAULT_EVAL_DIR = Path("EvalFin_101_cycle5_v4_cbarAC_potentialBaseline")
DEFAULT_SOFT_LABEL_DIR = Path("Data") / "assb_soft_labels_cycle5_v4"
DEFAULT_MODEL_DIR = Path("ModelFin_101")
DEFAULT_OUTPUT_SUBDIR = "surface_cs_cycle5_ID101_plotly"
DEFAULT_CYCLE_LABEL = "cycle5_v4"
DEFAULT_MODEL_LABEL = "ID101"
SCRIPT_VERSION = "2026-05-05-plotly-true-viridis-pred-blue-red"
DEFAULT_FONT_FAMILY = "Times New Roman"
DEFAULT_CMAP = "viridis"
DEFAULT_TRUE_CMAP = "viridis"
DEFAULT_PRED_CMAP = "coolwarm"
DEFAULT_TRUE_COLORSCALE = "Viridis"
DEFAULT_PRED_COLORSCALE = "RdBu_r"

# Larger visual blocks = fewer surface polygons = smoother real-time dragging.
DEFAULT_TIME_STRIDE = 5
DEFAULT_R_STRIDE = 2
DEFAULT_MAX_T_POINTS = 220
DEFAULT_MAX_R_POINTS = 40


class ArrayStore:
    """Small helper that stores arrays from one or more npz files with source names."""

    def __init__(self) -> None:
        self.items: Dict[str, np.ndarray] = {}
        self.sources: Dict[str, str] = {}

    def add_npz(self, path: Path, prefix: Optional[str] = None) -> None:
        if not path.exists():
            return
        try:
            data = np.load(path, allow_pickle=True)
        except Exception as exc:
            print(f"[WARN] Could not read npz: {path} ({exc})")
            return
        stem = prefix if prefix is not None else path.stem
        for key in data.files:
            arr = np.asarray(data[key])
            # Store both a source-qualified key and, if free, a plain key.
            qkey = f"{stem}:{key}"
            self.items[qkey] = arr
            self.sources[qkey] = str(path)
            if key not in self.items:
                self.items[key] = arr
                self.sources[key] = str(path)

    def keys(self) -> List[str]:
        return list(self.items.keys())

    def get(self, key: str) -> np.ndarray:
        return self.items[key]

    def source(self, key: str) -> str:
        return self.sources.get(key, "")


def _norm_name(name: str) -> str:
    """Normalize names for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _is_numeric_array(arr: np.ndarray) -> bool:
    return arr.size > 0 and np.issubdtype(arr.dtype, np.number)


def _squeeze_numeric(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim > 0:
        arr = np.squeeze(arr)
    return arr.astype(float, copy=False)


def find_first(store: ArrayStore, aliases: Sequence[str], *, min_ndim: int = 1, max_ndim: int = 2) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Exact/fuzzy lookup by aliases."""
    norm_aliases = [_norm_name(a) for a in aliases]

    # Exact key first.
    for key in store.keys():
        short = key.split(":")[-1]
        if short in aliases or key in aliases:
            arr = store.get(key)
            if _is_numeric_array(arr) and min_ndim <= np.squeeze(arr).ndim <= max_ndim:
                return key, _squeeze_numeric(arr)

    # Normalized/fuzzy key.
    for key in store.keys():
        arr = store.get(key)
        if not _is_numeric_array(arr):
            continue
        arr_sq = np.squeeze(arr)
        if not (min_ndim <= arr_sq.ndim <= max_ndim):
            continue
        nk = _norm_name(key)
        # Avoid over-broad fuzzy matches such as alias "ra" matching "x_train"/"r_train".
        # Two-letter axis aliases are only accepted as exact normalized matches.
        ok = False
        for a in norm_aliases:
            if a == nk:
                ok = True
                break
            if len(a) >= 3 and a in nk:
                ok = True
                break
        if ok:
            return key, _squeeze_numeric(arr)

    return None, None


def collect_eval_arrays(eval_dir: Path, eval_npz: Optional[Path] = None) -> ArrayStore:
    store = ArrayStore()
    if eval_npz is not None:
        store.add_npz(eval_npz, prefix=eval_npz.stem)
        return store

    if not eval_dir.exists():
        return store

    # Prefer top-level npz files, then one-level nested npz files.
    npz_files = sorted(eval_dir.glob("*.npz"))
    npz_files += sorted(p for p in eval_dir.glob("*/*.npz") if p.is_file())

    # Load likely array files first, so plain keys come from these files.
    def priority(p: Path) -> int:
        n = _norm_name(p.name)
        score = 100
        for token, s in [
            ("eval", 0), ("pred", 0), ("prediction", 0), ("array", 0),
            ("surface", 1), ("result", 1), ("debug", 10),
            ("data", 20), ("loss", 50),
        ]:
            if token in n:
                score = min(score, s)
        return score

    for path in sorted(npz_files, key=priority):
        store.add_npz(path, prefix=path.stem)
    return store


def collect_soft_arrays(soft_label_dir: Path) -> ArrayStore:
    store = ArrayStore()
    if not soft_label_dir.exists():
        return store
    # solution.npz is the preferred source because it already has 2D cs_a/cs_c.
    store.add_npz(soft_label_dir / "solution.npz", prefix="soft_solution")
    for name in ["data_cs_a.npz", "data_cs_c.npz", "data_phie.npz", "data_phis_c.npz"]:
        store.add_npz(soft_label_dir / name, prefix=Path(name).stem)
    return store


def merge_stores(primary: ArrayStore, secondary: ArrayStore) -> ArrayStore:
    out = ArrayStore()
    # Preserve insertion-like order: primary first, then secondary.
    for src in [primary, secondary]:
        for k in src.keys():
            if k not in out.items:
                out.items[k] = src.items[k]
                out.sources[k] = src.sources.get(k, "")
    return out


def _axis_from_exact(store: ArrayStore, keys: Sequence[str]) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Read an exact axis key before fuzzy matching. This prevents flattened eval grids
    such as r_a_grid with length nt*nr from being mistaken for the 1D radial vector."""
    for key in keys:
        if key in store.items:
            arr = store.get(key)
            if _is_numeric_array(arr):
                return key, _squeeze_numeric(arr)
    return None, None


def _axis_vector(arr: np.ndarray) -> np.ndarray:
    """Convert either a true 1D axis or a repeated flattened mesh axis to a 1D vector."""
    x = np.ravel(_squeeze_numeric(arr))
    if x.size == 0:
        return x
    # If the array is a flattened mesh axis, it contains many repeated values.
    # np.unique is safe for time/radial axes because both are monotonic grids.
    u = np.unique(x)
    if 1 < u.size < x.size:
        return u.astype(float, copy=False)
    return x.astype(float, copy=False)


def _collapse_axis_array(arr: np.ndarray, *, axis_name: str, nt: Optional[int] = None) -> np.ndarray:
    """Return a 1D physical axis from either a pure axis vector or a flattened mesh column.

    Some EvalFin npz files save r_a/r_c as one value per (t,r) sample, e.g. length
    nt*nr = 925*64 = 59200.  Matplotlib needs only the unique radial axis length nr.
    """
    a = _squeeze_numeric(arr)
    if a.ndim > 1:
        a = np.ravel(a)
    else:
        a = np.ravel(a)
    a = a[np.isfinite(a)]
    if a.size == 0:
        raise RuntimeError(f"Empty {axis_name} axis array.")

    # If this is a flattened time/radius mesh column, recover the compact axis.
    if nt is not None and nt > 0 and a.size > nt and a.size % nt == 0:
        nr = a.size // nt
        grid = a.reshape(nt, nr)
        # Radius is usually repeated for every time row.
        if np.allclose(grid, grid[0:1, :], rtol=1e-8, atol=1e-12):
            return np.asarray(grid[0, :], dtype=float)
        # Time is usually repeated for every radius column.
        if np.allclose(grid, grid[:, 0:1], rtol=1e-8, atol=1e-12):
            return np.asarray(grid[:, 0], dtype=float)

    # Fall back to unique values for repeated mesh columns.
    uniq = np.unique(a)
    # Keep dense true time axis as-is if it was not repeated. For radial axes, unique is safe.
    if axis_name.lower().startswith("r") and uniq.size < a.size:
        return uniq.astype(float, copy=False)
    if axis_name.lower().startswith("t") and uniq.size < a.size and (nt is None or uniq.size <= nt):
        return uniq.astype(float, copy=False)
    return a.astype(float, copy=False)


def infer_t_and_r(eval_store: ArrayStore, soft_store: ArrayStore) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, str]]:
    # Prefer soft-label solution.npz for physical axes. EvalFin may contain flattened mesh
    # columns named r_a/r_c with length nt*nr, which caused the previous 59200-radius bug.
    combined = merge_stores(soft_store, eval_store)
    src: Dict[str, str] = {}

    t_aliases = [
        "soft_solution:t", "solution:t", "t", "time", "times", "time_s", "t_s", "t_eval", "t_grid", "t_true",
    ]
    ra_aliases = [
        "soft_solution:r_a", "solution:r_a", "r_a", "ra", "r_anode", "r_negative", "r_neg", "r_a_m", "r_grid_a", "r_cs_a",
    ]
    rc_aliases = [
        "soft_solution:r_c", "solution:r_c", "r_c", "rc", "r_cathode", "r_positive", "r_pos", "r_c_m", "r_grid_c", "r_cs_c",
    ]

    kt, t_raw = find_first(combined, t_aliases, min_ndim=1, max_ndim=2)
    if t_raw is None:
        raise RuntimeError("Could not find time axis t. Expected solution.npz key 't' or equivalent.")
    t = _collapse_axis_array(t_raw, axis_name="t", nt=None)
    nt = int(t.size)

    kra, r_a_raw = find_first(combined, ra_aliases, min_ndim=1, max_ndim=2)
    krc, r_c_raw = find_first(combined, rc_aliases, min_ndim=1, max_ndim=2)

    if r_a_raw is None or r_c_raw is None:
        raise RuntimeError(
            "Could not find r_a/r_c arrays. Expected solution.npz with keys t, r_a, r_c "
            "or equivalent arrays in the evaluation npz."
        )

    r_a = _collapse_axis_array(r_a_raw, axis_name="r_a", nt=nt)
    r_c = _collapse_axis_array(r_c_raw, axis_name="r_c", nt=nt)

    src["t"] = f"{kt} <- {combined.source(kt or '')}"
    src["r_a"] = f"{kra} <- {combined.source(kra or '')}"
    src["r_c"] = f"{krc} <- {combined.source(krc or '')}"
    return np.ravel(t), np.ravel(r_a), np.ravel(r_c), src


def _reshape_if_flat(arr: np.ndarray, nt: int, nr: int, x_grid: Optional[np.ndarray] = None) -> np.ndarray:
    arr = _squeeze_numeric(arr)
    if arr.ndim == 2 and arr.shape == (nt, nr):
        return arr
    if arr.ndim == 2 and arr.shape == (nr, nt):
        return arr.T
    if arr.ndim == 1 and arr.size == nt * nr:
        # The soft-label data_cs_*.npz files are ordered as time-major, radius-minor.
        return arr.reshape(nt, nr)
    if arr.ndim == 2 and 1 in arr.shape and arr.size == nt * nr:
        return arr.reshape(nt, nr)

    # Reconstruct from x_train=(t,r) if available. This is a fallback for nonstandard eval outputs.
    if x_grid is not None:
        x = np.asarray(x_grid)
        y = np.ravel(arr)
        if x.ndim == 2 and x.shape[0] == y.size and x.shape[1] >= 2:
            t_vals = np.unique(x[:, 0])
            r_vals = np.unique(x[:, 1])
            if t_vals.size == nt and r_vals.size == nr and y.size == nt * nr:
                order = np.lexsort((x[:, 1], x[:, 0]))
                return y[order].reshape(nt, nr)

    raise RuntimeError(f"Array shape {arr.shape} cannot be interpreted as a ({nt}, {nr}) surface.")


def _score_surface_key(key: str, variable: str, role: str) -> int:
    """Higher score means a key is more likely to match variable+role."""
    n = _norm_name(key)
    var_tokens = {
        "cs_a": ["csa", "csan", "csa", "csa", "csa", "csa", "csa", "csa", "csa", "csa", "csa", "csa", "csa", "csneg", "csnegative", "csanode", "csa"],
        "cs_c": ["csc", "csca", "cscathode", "cspositive", "cspos", "csc"],
    }[variable]
    # Strong aliases that preserve underscore are normalized above.
    if variable == "cs_a":
        var_tokens += ["csa", "cs_a".replace("_", ""), "csan", "csanode"]
    else:
        var_tokens += ["csc", "cs_c".replace("_", ""), "csca", "cscathode"]

    true_tokens = ["true", "label", "soft", "target", "ref", "reference", "pde", "gt", "ytrue", "truth"]
    pred_tokens = ["pred", "prediction", "pinn", "model", "hat", "ypred", "nn", "out"]
    bad_for_pred = ["true", "label", "soft", "target", "ref", "reference", "pde", "gt", "truth"]
    bad_for_true = ["pred", "prediction", "pinn", "model", "hat", "ypred", "nn"]

    score = 0
    if any(tok in n for tok in var_tokens):
        score += 100
    else:
        return -999

    if role == "true":
        if any(tok in n for tok in true_tokens):
            score += 60
        if any(tok in n for tok in bad_for_true):
            score -= 80
        # soft_solution:cs_a or plain solution cs_a is usually true.
        if "softsolution" in n or "solution" in n:
            score += 30
    else:
        if any(tok in n for tok in pred_tokens):
            score += 60
        if any(tok in n for tok in bad_for_pred):
            score -= 80
        # Avoid choosing soft solution as prediction.
        if "softsolution" in n or "datacs" in n:
            score -= 120

    # Prefer concentration surfaces over x/y training vectors if both exist.
    if "xtrain" in n or "xparam" in n:
        score -= 150
    if "ytrain" in n:
        # y_train is true soft label, not prediction.
        score += 20 if role == "true" else -80
    return score


def find_surface(
    eval_store: ArrayStore,
    soft_store: ArrayStore,
    variable: str,
    role: str,
    nt: int,
    nr: int,
) -> Tuple[np.ndarray, str]:
    """Find cs_a/cs_c true or prediction surface."""
    assert variable in {"cs_a", "cs_c"}
    assert role in {"true", "pred"}

    combined = merge_stores(eval_store, soft_store) if role == "true" else eval_store

    candidates: List[Tuple[int, str, np.ndarray]] = []
    x_grid: Optional[np.ndarray] = None

    # Possible x grid for flattened reconstruction.
    x_aliases = [f"x_{variable}", f"x_train_{variable}", f"data_{variable}:x_train", "x_train"]
    _, x_grid = find_first(combined, x_aliases, min_ndim=2, max_ndim=2)

    for key in combined.keys():
        arr = combined.get(key)
        if not _is_numeric_array(arr):
            continue
        score = _score_surface_key(key, variable, role)
        # For predictions, avoid silently using soft-label data_cs_*:y_train as if it were a model output.
        # Plain eval keys such as "cs_a"/"cs_c" still pass because their score is 100;
        # explicit keys such as "cs_a_pred"/"pred_cs_a" score even higher.
        if score < (80 if role == "pred" else 0):
            continue
        arr_sq = np.squeeze(arr)
        # Accept 2D surfaces and flattened surfaces only.
        if arr_sq.ndim == 2 and (arr_sq.shape in [(nt, nr), (nr, nt)] or arr_sq.size == nt * nr):
            candidates.append((score + 10, key, arr_sq))
        elif arr_sq.ndim == 1 and arr_sq.size == nt * nr:
            candidates.append((score, key, arr_sq))

    # Direct soft-label fallback for truth.
    if role == "true":
        soft_key = f"soft_solution:{variable}"
        if soft_key in soft_store.items:
            arr = soft_store.get(soft_key)
            candidates.append((999, soft_key, arr))
        elif variable in soft_store.items:
            arr = soft_store.get(variable)
            candidates.append((900, variable, arr))

    if not candidates:
        role_msg = "prediction" if role == "pred" else "truth"
        available = "\n".join(f"  - {k}: {combined.get(k).shape}" for k in combined.keys()[:120])
        raise RuntimeError(
            f"Could not find {role_msg} surface for {variable}.\n"
            f"Need a full 2D array or flattened array with nt*nr={nt*nr}.\n"
            f"Available arrays include:\n{available}\n\n"
            f"Tip: re-run evaluate_assb_pinn_vs_softlabels.py and make sure it saves keys such as "
            f"{variable}_true and {variable}_pred, or pass --eval_npz explicitly."
        )

    candidates.sort(key=lambda z: z[0], reverse=True)
    last_error: Optional[Exception] = None
    for score, key, arr in candidates:
        try:
            surf = _reshape_if_flat(arr, nt, nr, x_grid=x_grid)
            return surf, f"{key} <- {combined.source(key)}"
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"Found candidate arrays for {variable}/{role}, but none could be reshaped. Last error: {last_error}")


def align_by_shape_and_time(
    t: np.ndarray,
    r: np.ndarray,
    true_z: np.ndarray,
    pred_z: np.ndarray,
    variable: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align arrays if prediction has slightly different time/radius resolution."""
    true_z = np.asarray(true_z, dtype=float)
    pred_z = np.asarray(pred_z, dtype=float)

    if true_z.shape == pred_z.shape:
        return t, r, true_z, pred_z

    # If one array is transposed, fix it.
    if true_z.shape == pred_z.T.shape:
        return t, r, true_z, pred_z.T

    raise RuntimeError(
        f"{variable}: true surface shape {true_z.shape} and predicted surface shape {pred_z.shape} do not match. "
        "This script intentionally avoids silent interpolation for metric computation. "
        "Re-run the evaluation so true/pred are saved on the same t-r grid."
    )


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.ravel(np.asarray(y_true, dtype=float))
    yp = np.ravel(np.asarray(y_pred, dtype=float))
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    if yt.size == 0:
        return {"mae": math.nan, "rmse": math.nan, "r2": math.nan, "nmae": math.nan, "nrmse": math.nan, "corr": math.nan, "range": math.nan}

    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    rng = float(np.max(yt) - np.min(yt))
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else math.nan
    nmae = float(mae / rng) if rng > 0 else math.nan
    nrmse = float(rmse / rng) if rng > 0 else math.nan
    if np.std(yt) > 0 and np.std(yp) > 0:
        corr = float(np.corrcoef(yt, yp)[0, 1])
    else:
        corr = math.nan
    return {"mae": mae, "rmse": rmse, "r2": r2, "nmae": nmae, "nrmse": nrmse, "corr": corr, "range": rng}


def fmt_metrics(m: Dict[str, float]) -> str:
    def f(x: float, nd: int = 6) -> str:
        return "nan" if not np.isfinite(x) else f"{x:.{nd}f}"

    def pct(x: float) -> str:
        return "nan" if not np.isfinite(x) else f"{100.0 * x:.3f}%"

    return f"R²={f(m['r2'])} | NMAE={pct(m['nmae'])} | NRMSE={pct(m['nrmse'])} | corr={f(m['corr'])}"


def choose_indices(n: int, stride: int, max_points: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("Cannot downsample empty dimension.")
    stride = max(1, int(stride))
    idx = np.arange(0, n, stride, dtype=int)
    if idx[-1] != n - 1:
        idx = np.r_[idx, n - 1]
    if max_points > 0 and idx.size > max_points:
        idx = np.unique(np.linspace(0, n - 1, max_points, dtype=int))
    return idx


def convert_time(t: np.ndarray, unit: str) -> Tuple[np.ndarray, str]:
    unit = unit.lower()
    if unit in {"s", "sec", "second", "seconds"}:
        return t, "time / s"
    if unit in {"min", "minute", "minutes"}:
        return t / 60.0, "time / min"
    if unit in {"h", "hr", "hour", "hours"}:
        return t / 3600.0, "time / h"
    raise ValueError(f"Unsupported time unit: {unit}")


def convert_r(r: np.ndarray, unit: str) -> Tuple[np.ndarray, str]:
    unit = unit.lower()
    if unit in {"m", "meter", "meters"}:
        return r, "radial r / m"
    if unit in {"um", "µm", "micron", "microns", "micrometer", "micrometers"}:
        return r * 1e6, "radial r / μm"
    if unit in {"nm", "nanometer", "nanometers"}:
        return r * 1e9, "radial r / nm"
    raise ValueError(f"Unsupported r unit: {unit}")


def setup_matplotlib_backend(backend: str, no_show: bool) -> None:
    import matplotlib

    if no_show:
        matplotlib.use("Agg", force=True)
        return

    if backend and backend.lower() != "auto":
        try:
            matplotlib.use(backend, force=True)
            return
        except Exception as exc:
            print(f"[WARN] Could not use backend {backend}: {exc}")

    # Auto preference for Windows-like interactive usage.
    for cand in ["TkAgg", "QtAgg", "Qt5Agg", "WXAgg"]:
        try:
            matplotlib.use(cand, force=True)
            return
        except Exception:
            continue
    print("[WARN] No preferred interactive backend found. Matplotlib will use its default backend.")


def resolve_font_family(requested: str = DEFAULT_FONT_FAMILY) -> str:
    """Return the requested font if Matplotlib can see it; otherwise use a safe serif fallback.

    On the target Windows machine, Times New Roman is normally available. The fallback only
    avoids noisy findfont warnings on machines where it is absent.
    """
    try:
        from matplotlib import font_manager
        available = {f.name for f in font_manager.fontManager.ttflist}
        for candidate in [requested, "Times New Roman", "Times", "DejaVu Serif"]:
            if candidate in available:
                if candidate != requested:
                    print(f"[WARN] Requested font {requested!r} was not found; using {candidate!r} instead.")
                return candidate
    except Exception:
        pass
    return requested


def setup_matplotlib_style(font_family: str = DEFAULT_FONT_FAMILY) -> None:
    """Apply Times New Roman style to all Matplotlib-rendered text."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": [font_family, "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "figure.titleweight": "normal",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })


def force_axis_font(ax, font_family: str = DEFAULT_FONT_FAMILY) -> None:
    """Best-effort font assignment for 3D axes, tick labels, labels, and title."""
    try:
        for artist in [ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label]:
            artist.set_fontname(font_family)
    except Exception:
        pass
    try:
        for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            label.set_fontname(font_family)
    except Exception:
        pass


def force_colorbar_font(cbar, font_family: str = DEFAULT_FONT_FAMILY) -> None:
    try:
        cbar.ax.yaxis.label.set_fontname(font_family)
        for label in cbar.ax.get_yticklabels():
            label.set_fontname(font_family)
    except Exception:
        pass


def force_figure_font(fig, font_family: str = DEFAULT_FONT_FAMILY) -> None:
    try:
        import matplotlib.text as mtext
        for text in fig.findobj(match=mtext.Text):
            text.set_fontname(font_family)
    except Exception:
        pass


def plot_one_surface(
    *,
    t: np.ndarray,
    r: np.ndarray,
    z: np.ndarray,
    variable: str,
    role: str,
    metrics: Dict[str, float],
    model_label: str,
    cycle_label: str,
    t_label: str,
    r_label: str,
    z_label: str,
    zlim: Tuple[float, float],
    cmap: str,
    save_path: Optional[Path],
    elev: float,
    azim: float,
    show_colorbar: bool,
    font_family: str = DEFAULT_FONT_FAMILY,
):
    import matplotlib.pyplot as plt
    from matplotlib import colors

    T, R = np.meshgrid(t, r, indexing="ij")
    fig = plt.figure(figsize=(10.5, 7.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    norm = colors.Normalize(vmin=zlim[0], vmax=zlim[1])
    surf = ax.plot_surface(
        T,
        R,
        z,
        cmap=cmap,
        norm=norm,
        linewidth=0.0,
        antialiased=False,
        shade=False,
        rstride=1,
        cstride=1,
    )

    title_role = "TRUE" if role == "true" else "PRED"
    top = f"{model_label} {cycle_label} | {variable} {title_role} | {fmt_metrics(metrics)}"
    ax.set_title(top, pad=20, fontsize=11)
    ax.set_xlabel(t_label, labelpad=10)
    ax.set_ylabel(r_label, labelpad=10)
    ax.set_zlabel(z_label, labelpad=10)
    ax.set_zlim(*zlim)
    ax.view_init(elev=elev, azim=azim)
    force_axis_font(ax, font_family)
    try:
        ax.set_box_aspect((1.8, 1.0, 0.8))
    except Exception:
        pass
    ax.grid(True)
    if show_colorbar:
        cbar = fig.colorbar(surf, ax=ax, shrink=0.68, pad=0.08, aspect=22)
        cbar.set_label(z_label)
        force_colorbar_font(cbar, font_family)
    force_figure_font(fig, font_family)

    try:
        fig.canvas.manager.set_window_title(f"{model_label} {cycle_label} {variable} {title_role}")
    except Exception:
        pass

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
        print(f"[SAVE] {save_path}")
    return fig, ax


def plot_grid(
    *,
    data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float], Tuple[float, float], str]],
    model_label: str,
    cycle_label: str,
    t_label: str,
    r_label_map: Dict[str, str],
    z_label: str,
    true_cmap: str,
    pred_cmap: str,
    save_path: Optional[Path],
    elev: float,
    azim: float,
    font_family: str = DEFAULT_FONT_FAMILY,
):
    """Optional 2x2 layout; separate figures are default because they drag faster."""
    import matplotlib.pyplot as plt
    from matplotlib import colors

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    order = [("cs_a", "true"), ("cs_a", "pred"), ("cs_c", "true"), ("cs_c", "pred")]
    for i, (variable, role) in enumerate(order, start=1):
        t, r, z, metrics, zlim, r_label = data[f"{variable}_{role}"]
        T, R = np.meshgrid(t, r, indexing="ij")
        ax = fig.add_subplot(2, 2, i, projection="3d")
        norm = colors.Normalize(vmin=zlim[0], vmax=zlim[1])
        role_cmap = true_cmap if role == "true" else pred_cmap
        surf = ax.plot_surface(T, R, z, cmap=role_cmap, norm=norm, linewidth=0, antialiased=False, shade=False)
        title_role = "TRUE" if role == "true" else "PRED"
        ax.set_title(f"{variable} {title_role}\n{fmt_metrics(metrics)}", fontsize=9, pad=10)
        ax.set_xlabel(t_label)
        ax.set_ylabel(r_label)
        ax.set_zlabel(z_label)
        ax.set_zlim(*zlim)
        ax.view_init(elev=elev, azim=azim)
        force_axis_font(ax, font_family)
        try:
            ax.set_box_aspect((1.8, 1.0, 0.8))
        except Exception:
            pass
        cbar = fig.colorbar(surf, ax=ax, shrink=0.58, pad=0.05, aspect=18)
        force_colorbar_font(cbar, font_family)
    fig.suptitle(f"{model_label} {cycle_label} | cs_a / cs_c surfaces", fontsize=14, fontname=font_family)
    force_figure_font(fig, font_family)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
        print(f"[SAVE] {save_path}")
    return fig


def save_metrics_json(path: Path, metrics: Dict[str, Dict[str, float]], sources: Dict[str, str], args: argparse.Namespace) -> None:
    payload: Dict[str, Any] = {
        "model_label": args.model_label,
        "cycle_label": args.cycle_label,
        "metrics": metrics,
        "sources": sources,
        "normalization": "NMAE and NRMSE are normalized by max(true)-min(true) over the full t-r surface.",
        "plot_style": {
            "font_family": getattr(args, "font_family", DEFAULT_FONT_FAMILY),
            "cmap": getattr(args, "cmap", DEFAULT_CMAP),
            "true_cmap": getattr(args, "true_cmap", None) or DEFAULT_TRUE_CMAP,
            "pred_cmap": getattr(args, "pred_cmap", None) or DEFAULT_PRED_CMAP,
        },
        "plot_downsampling": {
            "time_stride": args.time_stride,
            "r_stride": args.r_stride,
            "max_t_points": args.max_t_points,
            "max_r_points": args.max_r_points,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {path}")



def predict_surfaces_from_model(
    *,
    repo_root: Path,
    model_dir: Path,
    soft_label_dir: Path,
    checkpoint: Optional[Path],
    ocp_dir: Optional[Path],
    batch_size: int,
    nt: int,
    nr_a: int,
    nr_c: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    """Fallback path: evaluate ModelFin_101 directly if EvalFin has no saved prediction npz.

    This reuses the project evaluator functions, so it stays aligned with the current
    model-loading, per-electrode radial normalization, output index order, and rescale logic.
    """
    repo_root = repo_root.resolve()
    model_dir = model_dir.resolve()
    soft_label_dir = soft_label_dir.resolve()
    if ocp_dir is not None:
        os.environ["ASSB_OCP_DIR"] = str(ocp_dir.resolve())
    os.environ["ASSB_SOFT_LABEL_DIR"] = str(soft_label_dir)

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    util_dir = repo_root / "util"
    if str(util_dir) not in sys.path:
        sys.path.insert(0, str(util_dir))

    try:
        import evaluate_assb_pinn_vs_softlabels as ev  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Prediction arrays were not found in EvalFin, and the fallback could not import "
            "evaluate_assb_pinn_vs_softlabels.py from the project root. "
            "Put this script in the project root or pass --eval_npz containing cs_a_pred/cs_c_pred. "
            f"Import error: {exc}"
        ) from exc

    if checkpoint is None:
        ckpt = ev._find_checkpoint(model_dir, None)
    else:
        ckpt = checkpoint if checkpoint.is_absolute() else model_dir / checkpoint
        ckpt = ckpt.resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print("\n[FALLBACK] EvalFin prediction arrays were not found. Recomputing cs_a/cs_c from the model.")
    print(f"[FALLBACK] model_dir  = {model_dir}")
    print(f"[FALLBACK] checkpoint = {ckpt}")
    print(f"[FALLBACK] batch_size = {batch_size}")

    nn, _config, chosen_summary = ev.load_model_for_eval(
        repo_root=repo_root,
        model_dir=model_dir,
        soft_label_dir=soft_label_dir,
        checkpoint=ckpt,
        force_summary_from_soft_labels=True,
    )

    pred_sources: Dict[str, str] = {
        "model_dir": str(model_dir),
        "checkpoint": str(ckpt),
        "chosen_train_summary_json": str(chosen_summary),
    }

    preds: Dict[str, np.ndarray] = {}
    for var, nr, fn in [("cs_a", nr_a, "data_cs_a.npz"), ("cs_c", nr_c, "data_cs_c.npz")]:
        data_path = soft_label_dir / fn
        data = ev._load_npz(data_path)
        for key in ["x_train", "y_train", "x_params_train"]:
            if key not in data:
                raise KeyError(f"{data_path} is missing key {key}")
        pred_flat, _debug = ev.predict_dataset(
            nn=nn,
            variable=var,
            x=data["x_train"],
            x_params=data["x_params_train"],
            batch_size=batch_size,
            debug_first=False,
        )
        preds[var] = _reshape_if_flat(pred_flat, nt, nr, x_grid=data.get("x_train"))
        pred_sources[f"{var}_pred"] = f"computed from {ckpt} using {data_path}"

    return preds["cs_a"], preds["cs_c"], pred_sources



# -----------------------------
# Plotly output helpers
# -----------------------------

def _plotly_colorscale(name: str, *, role: str) -> str:
    """Map common Matplotlib colormap names to Plotly colorscale names."""
    if not name:
        return DEFAULT_TRUE_COLORSCALE if role == "true" else DEFAULT_PRED_COLORSCALE
    key = str(name).strip()
    low = key.lower()
    aliases = {
        "viridis": "Viridis",
        "plasma": "Plasma",
        "cividis": "Cividis",
        "magma": "Magma",
        "inferno": "Inferno",
        "coolwarm": "RdBu_r",  # low blue -> high red, closest Plotly built-in to Matplotlib coolwarm
        "rdbu_r": "RdBu_r",
        "rdbu": "RdBu",
        "bluered": "RdBu_r",
        "blue-red": "RdBu_r",
        "blue_to_red": "RdBu_r",
    }
    return aliases.get(low, key)


def plotly_surface_figure(
    *,
    t: np.ndarray,
    r: np.ndarray,
    z: np.ndarray,
    variable: str,
    role: str,
    metrics: Dict[str, float],
    model_label: str,
    cycle_label: str,
    t_label: str,
    r_label: str,
    z_label: str,
    zlim: Tuple[float, float],
    colorscale: str,
    font_family: str,
):
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        raise RuntimeError(
            "Plotly is required for this script. Install it in the torchgpu environment, e.g.\n"
            "D:\\Anaconda\\envs\\torchgpu\\python.exe -m pip install plotly"
        ) from exc

    T, R = np.meshgrid(t, r, indexing="ij")
    title_role = "TRUE" if role == "true" else "PRED"
    title = f"{model_label} {cycle_label} | {variable} {title_role} | {fmt_metrics(metrics)}"

    surf = go.Surface(
        x=T,
        y=R,
        z=z,
        surfacecolor=z,
        colorscale=colorscale,
        cmin=zlim[0],
        cmax=zlim[1],
        showscale=True,
        colorbar=dict(
            title=dict(text=z_label, font=dict(family=font_family, size=14)),
            tickfont=dict(family=font_family, size=12),
            len=0.72,
            thickness=22,
        ),
        hovertemplate=(
            f"{t_label}: %{{x:.6g}}<br>"
            f"{r_label}: %{{y:.6g}}<br>"
            f"{z_label}: %{{z:.6g}}<extra>{variable} {title_role}</extra>"
        ),
        lighting=dict(ambient=0.75, diffuse=0.85, roughness=0.8, specular=0.05),
        lightposition=dict(x=100, y=-200, z=200),
    )

    fig = go.Figure(data=[surf])
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(family=font_family, size=18)),
        font=dict(family=font_family, size=14),
        margin=dict(l=10, r=10, b=10, t=70),
        width=1150,
        height=820,
        scene=dict(
            xaxis=dict(title=dict(text=t_label, font=dict(family=font_family, size=14)), tickfont=dict(family=font_family, size=12)),
            yaxis=dict(title=dict(text=r_label, font=dict(family=font_family, size=14)), tickfont=dict(family=font_family, size=12)),
            zaxis=dict(title=dict(text=z_label, font=dict(family=font_family, size=14)), tickfont=dict(family=font_family, size=12), range=[zlim[0], zlim[1]]),
            aspectmode="manual",
            aspectratio=dict(x=1.8, y=1.0, z=0.8),
            camera=dict(eye=dict(x=1.65, y=-1.65, z=0.95)),
        ),
    )
    return fig


def write_plotly_surface(
    *,
    t: np.ndarray,
    r: np.ndarray,
    z: np.ndarray,
    variable: str,
    role: str,
    metrics: Dict[str, float],
    model_label: str,
    cycle_label: str,
    t_label: str,
    r_label: str,
    z_label: str,
    zlim: Tuple[float, float],
    colorscale: str,
    output_path: Path,
    font_family: str,
    auto_open: bool,
) -> Path:
    fig = plotly_surface_figure(
        t=t,
        r=r,
        z=z,
        variable=variable,
        role=role,
        metrics=metrics,
        model_label=model_label,
        cycle_label=cycle_label,
        t_label=t_label,
        r_label=r_label,
        z_label=z_label,
        zlim=zlim,
        colorscale=colorscale,
        font_family=font_family,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True, auto_open=auto_open)
    print(f"[SAVE] {output_path}")
    return output_path


def save_plotly_metrics_json(path: Path, metrics: Dict[str, Dict[str, float]], sources: Dict[str, str], args: argparse.Namespace) -> None:
    payload: Dict[str, Any] = {
        "model_label": args.model_label,
        "cycle_label": args.cycle_label,
        "metrics": metrics,
        "sources": sources,
        "normalization": "NMAE and NRMSE are normalized by max(true)-min(true) over the full t-r surface.",
        "plot_style": {
            "font_family": args.font_family,
            "true_colorscale": args.true_colorscale or DEFAULT_TRUE_COLORSCALE,
            "pred_colorscale": args.pred_colorscale or DEFAULT_PRED_COLORSCALE,
        },
        "plot_downsampling": {
            "time_stride": args.time_stride,
            "r_stride": args.r_stride,
            "max_t_points": args.max_t_points,
            "max_r_points": args.max_r_points,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Plotly 3D surfaces for ID101 cycle5 cs_a/cs_c true/pred.")
    parser.add_argument("--eval_dir", type=Path, default=DEFAULT_EVAL_DIR, help="Evaluation output directory. Default: %(default)s")
    parser.add_argument("--soft_label_dir", type=Path, default=DEFAULT_SOFT_LABEL_DIR, help="Soft-label directory. Default: %(default)s")
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR, help="Model directory used by fallback prediction. Default: %(default)s")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint for fallback prediction, e.g. best.pt or ModelFin_101/best.pt.")
    parser.add_argument("--ocp_dir", type=Path, default=None, help="Optional OCP dir exported as ASSB_OCP_DIR before fallback prediction.")
    parser.add_argument("--eval_npz", type=Path, default=None, help="Optional explicit npz containing predicted arrays.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory for optional HTML and metrics JSON outputs.")
    parser.add_argument("--model_label", default=DEFAULT_MODEL_LABEL, help="Model label printed in titles. Default: %(default)s")
    parser.add_argument("--cycle_label", default=DEFAULT_CYCLE_LABEL, help="Cycle/data label printed in titles. Default: %(default)s")
    parser.add_argument("--backend", default="auto", help="Ignored; kept only for command compatibility with the Matplotlib script.")
    parser.add_argument("--time_stride", type=int, default=DEFAULT_TIME_STRIDE, help="Plot every Nth time point. Default: %(default)s")
    parser.add_argument("--r_stride", type=int, default=DEFAULT_R_STRIDE, help="Plot every Nth radial point. Default: %(default)s")
    parser.add_argument("--max_t_points", type=int, default=DEFAULT_MAX_T_POINTS, help="Max plotted time points. Default: %(default)s")
    parser.add_argument("--max_r_points", type=int, default=DEFAULT_MAX_R_POINTS, help="Max plotted radial points. Default: %(default)s")
    parser.add_argument("--time_unit", choices=["s", "min", "h"], default="s", help="Time axis unit. Default: %(default)s")
    parser.add_argument("--r_unit", choices=["m", "um", "nm"], default="um", help="Radial axis unit. Default: %(default)s")
    parser.add_argument("--true_colorscale", default=None, help="Plotly colorscale for truth surfaces. Default: Viridis")
    parser.add_argument("--pred_colorscale", default=None, help="Plotly colorscale for prediction surfaces. Default: RdBu_r")
    parser.add_argument("--font_family", default=DEFAULT_FONT_FAMILY, help="Font family used for all Plotly figure text. Default: %(default)s")
    parser.add_argument("--no_open", action="store_true", help="Save HTML files but do not open them in the browser.")
    parser.add_argument("--no_show", action="store_true", help="Alias of --no_open for compatibility with the Matplotlib script.")
    parser.add_argument("--no_model_fallback", action="store_true", help="If EvalFin lacks prediction arrays, fail instead of recomputing predictions from ModelFin_101.")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size for model fallback prediction. Default: %(default)s")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    script_root = Path(__file__).resolve().parent
    eval_dir = args.eval_dir if args.eval_dir.is_absolute() else script_root / args.eval_dir
    soft_label_dir = args.soft_label_dir if args.soft_label_dir.is_absolute() else script_root / args.soft_label_dir
    model_dir = args.model_dir if args.model_dir.is_absolute() else script_root / args.model_dir
    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = args.checkpoint if args.checkpoint.is_absolute() else script_root / args.checkpoint
    ocp_dir = None
    if args.ocp_dir is not None:
        ocp_dir = args.ocp_dir if args.ocp_dir.is_absolute() else script_root / args.ocp_dir
    eval_npz = None
    if args.eval_npz is not None:
        eval_npz = args.eval_npz if args.eval_npz.is_absolute() else script_root / args.eval_npz
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = eval_dir / DEFAULT_OUTPUT_SUBDIR
    elif not output_dir.is_absolute():
        output_dir = script_root / output_dir

    font_family = args.font_family
    true_colorscale = _plotly_colorscale(args.true_colorscale or DEFAULT_TRUE_COLORSCALE, role="true")
    pred_colorscale = _plotly_colorscale(args.pred_colorscale or DEFAULT_PRED_COLORSCALE, role="pred")

    eval_store = collect_eval_arrays(eval_dir, eval_npz=eval_npz)
    soft_store = collect_soft_arrays(soft_label_dir)

    print(f"[INFO] script version  = {SCRIPT_VERSION}")
    print(f"[INFO] eval_dir        = {eval_dir}")
    print(f"[INFO] soft_label_dir  = {soft_label_dir}")
    print(f"[INFO] font family     = {font_family}")
    print(f"[INFO] true colorscale = {true_colorscale}")
    print(f"[INFO] pred colorscale = {pred_colorscale}")
    print(f"[INFO] eval arrays     = {len(eval_store.keys())}")
    print(f"[INFO] soft arrays     = {len(soft_store.keys())}")

    t, r_a, r_c, sources = infer_t_and_r(eval_store, soft_store)
    nt = int(t.size)
    nr_a = int(r_a.size)
    nr_c = int(r_c.size)

    cs_a_true, src_a_true = find_surface(eval_store, soft_store, "cs_a", "true", nt, nr_a)
    cs_c_true, src_c_true = find_surface(eval_store, soft_store, "cs_c", "true", nt, nr_c)

    pred_source_extra: Dict[str, str] = {}
    try:
        cs_a_pred, src_a_pred = find_surface(eval_store, soft_store, "cs_a", "pred", nt, nr_a)
        cs_c_pred, src_c_pred = find_surface(eval_store, soft_store, "cs_c", "pred", nt, nr_c)
    except RuntimeError:
        if args.no_model_fallback:
            raise
        cs_a_pred, cs_c_pred, pred_source_extra = predict_surfaces_from_model(
            repo_root=script_root,
            model_dir=model_dir,
            soft_label_dir=soft_label_dir,
            checkpoint=checkpoint,
            ocp_dir=ocp_dir,
            batch_size=args.batch_size,
            nt=nt,
            nr_a=nr_a,
            nr_c=nr_c,
        )
        src_a_pred = pred_source_extra.get("cs_a_pred", "computed from model fallback")
        src_c_pred = pred_source_extra.get("cs_c_pred", "computed from model fallback")

    t_a, r_a, cs_a_true, cs_a_pred = align_by_shape_and_time(t, r_a, cs_a_true, cs_a_pred, "cs_a")
    t_c, r_c, cs_c_true, cs_c_pred = align_by_shape_and_time(t, r_c, cs_c_true, cs_c_pred, "cs_c")

    sources.update({
        "cs_a_true": src_a_true,
        "cs_a_pred": src_a_pred,
        "cs_c_true": src_c_true,
        "cs_c_pred": src_c_pred,
    })
    sources.update(pred_source_extra)

    metrics = {
        "cs_a": metric_dict(cs_a_true, cs_a_pred),
        "cs_c": metric_dict(cs_c_true, cs_c_pred),
    }

    print("\n[METRICS] computed on full-resolution surfaces")
    for var, m in metrics.items():
        print(
            f"  {var}: MAE={m['mae']:.8g}, RMSE={m['rmse']:.8g}, "
            f"R2={m['r2']:.8g}, NMAE={100*m['nmae']:.4g}%, "
            f"NRMSE={100*m['nrmse']:.4g}%, corr={m['corr']:.8g}, range={m['range']:.8g}"
        )

    print("\n[SOURCES]")
    for k, v in sources.items():
        print(f"  {k}: {v}")

    tidx_a = choose_indices(t_a.size, args.time_stride, args.max_t_points)
    ridx_a = choose_indices(r_a.size, args.r_stride, args.max_r_points)
    tidx_c = choose_indices(t_c.size, args.time_stride, args.max_t_points)
    ridx_c = choose_indices(r_c.size, args.r_stride, args.max_r_points)

    t_a_plot_raw = t_a[tidx_a]
    r_a_plot_raw = r_a[ridx_a]
    t_c_plot_raw = t_c[tidx_c]
    r_c_plot_raw = r_c[ridx_c]

    t_a_plot, t_label = convert_time(t_a_plot_raw, args.time_unit)
    t_c_plot, _ = convert_time(t_c_plot_raw, args.time_unit)
    r_a_plot, r_a_label = convert_r(r_a_plot_raw, args.r_unit)
    r_c_plot, r_c_label = convert_r(r_c_plot_raw, args.r_unit)

    z_a_true = cs_a_true[np.ix_(tidx_a, ridx_a)]
    z_a_pred = cs_a_pred[np.ix_(tidx_a, ridx_a)]
    z_c_true = cs_c_true[np.ix_(tidx_c, ridx_c)]
    z_c_pred = cs_c_pred[np.ix_(tidx_c, ridx_c)]

    zlim_a = (float(np.nanmin([np.nanmin(cs_a_true), np.nanmin(cs_a_pred)])), float(np.nanmax([np.nanmax(cs_a_true), np.nanmax(cs_a_pred)])))
    zlim_c = (float(np.nanmin([np.nanmin(cs_c_true), np.nanmin(cs_c_pred)])), float(np.nanmax([np.nanmax(cs_c_true), np.nanmax(cs_c_pred)])))

    save_plotly_metrics_json(output_dir / "cs_surface_metrics_ID101_cycle5_plotly.json", metrics, sources, args)

    print("\n[PLOT] Downsampled plotted grids")
    print(f"  cs_a: time {tidx_a.size}/{t_a.size}, radius {ridx_a.size}/{r_a.size}, surface cells ≈ {(tidx_a.size-1)*(ridx_a.size-1)}")
    print(f"  cs_c: time {tidx_c.size}/{t_c.size}, radius {ridx_c.size}/{r_c.size}, surface cells ≈ {(tidx_c.size-1)*(ridx_c.size-1)}")

    auto_open = not (args.no_open or args.no_show)
    files = []
    files.append(write_plotly_surface(
        t=t_a_plot, r=r_a_plot, z=z_a_true, variable="cs_a", role="true", metrics=metrics["cs_a"],
        model_label=args.model_label, cycle_label=args.cycle_label, t_label=t_label, r_label=r_a_label,
        z_label="concentration cs_a", zlim=zlim_a, colorscale=true_colorscale,
        output_path=output_dir / "cs_a_true_surface_plotly.html", font_family=font_family, auto_open=auto_open,
    ))
    files.append(write_plotly_surface(
        t=t_a_plot, r=r_a_plot, z=z_a_pred, variable="cs_a", role="pred", metrics=metrics["cs_a"],
        model_label=args.model_label, cycle_label=args.cycle_label, t_label=t_label, r_label=r_a_label,
        z_label="concentration cs_a", zlim=zlim_a, colorscale=pred_colorscale,
        output_path=output_dir / "cs_a_pred_surface_plotly.html", font_family=font_family, auto_open=auto_open,
    ))
    files.append(write_plotly_surface(
        t=t_c_plot, r=r_c_plot, z=z_c_true, variable="cs_c", role="true", metrics=metrics["cs_c"],
        model_label=args.model_label, cycle_label=args.cycle_label, t_label=t_label, r_label=r_c_label,
        z_label="concentration cs_c", zlim=zlim_c, colorscale=true_colorscale,
        output_path=output_dir / "cs_c_true_surface_plotly.html", font_family=font_family, auto_open=auto_open,
    ))
    files.append(write_plotly_surface(
        t=t_c_plot, r=r_c_plot, z=z_c_pred, variable="cs_c", role="pred", metrics=metrics["cs_c"],
        model_label=args.model_label, cycle_label=args.cycle_label, t_label=t_label, r_label=r_c_label,
        z_label="concentration cs_c", zlim=zlim_c, colorscale=pred_colorscale,
        output_path=output_dir / "cs_c_pred_surface_plotly.html", font_family=font_family, auto_open=auto_open,
    ))

    print("\n[INFO] Plotly HTML files generated:")
    for f in files:
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise
