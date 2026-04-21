# -*- coding: utf-8 -*-
"""
Plot 3D truth concentration surfaces from CSV (COMSOL truth, long-table format supported):
  - cs_a(t, r): anode solid lithium concentration
  - cs_c(t, r): cathode solid lithium concentration

NEW CSV paths (v2):
  C:\\Users\\Tiga_QJW\\Desktop\\cs_a_truth_5500t_x55r_v2.csv
  C:\\Users\\Tiga_QJW\\Desktop\\cs_c_truth_5500t_x55r_v2.csv

Style:
  - Times New Roman + bold
  - Nature palette (#403990, #80A6E2, #FBDD85, #F46F43, #CF3D3E)
  - Smaller surface patches:
      * plot_surface rcount/ccount refined
      * r-direction interpolation (important for 55-r)

Default behavior:
  - Save PNGs to ./Figures3D_Truth_v2/
  - ALSO show interactive windows (no extra args needed)

If you do NOT want to show windows:
  python plot3DTruthConcentrationSurface.py --no_show
"""

import os
import re
import argparse
import numpy as np


# -----------------------------
# Default CSV paths (v2 files)
# -----------------------------
DEFAULT_CS_A = r"C:\Users\Tiga_QJW\Desktop\cs_a_truth_5500t_x55r_v2.csv"
DEFAULT_CS_C = r"C:\Users\Tiga_QJW\Desktop\cs_c_truth_5500t_x55r_v2.csv"


# -----------------------------
# Matplotlib backend + style
# -----------------------------
def setup_matplotlib(show: bool):
    """
    Set backend BEFORE importing pyplot.
    show=True  -> interactive backend (Qt5Agg preferred)
    show=False -> Agg (save only)
    """
    import matplotlib
    if show:
        try:
            import PyQt5  # noqa: F401
            matplotlib.use("Qt5Agg")
        except Exception:
            matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.weight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.2,
        "mathtext.fontset": "stix",
    })


def get_nature_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#403990", "#80A6E2", "#FBDD85", "#F46F43", "#CF3D3E"]
    return LinearSegmentedColormap.from_list("nature_palette", colors, N=256)


# -----------------------------
# CSV -> (t_vec, r_vec, Z)  (supports long-table format)
# -----------------------------
def _pick_col(df_cols, patterns):
    """Pick first column whose lowercase name matches any regex in patterns."""
    for c in df_cols:
        cl = str(c).lower()
        for pat in patterns:
            if re.search(pat, cl):
                return c
    return None


def load_truth_surface_csv(path: str):
    """
    Return:
      t_vec: (Nt,)
      r_vec: (Nr,)
      Z:     (Nt, Nr)

    Supports:
      - Long table: columns include time + radius + value -> pivot to matrix
      - Wide table: first col time, remaining columns are r bins (if any)
      - Matrix fallback
    """
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError(
            "This script requires pandas to pivot your long-table CSV.\n"
            "Please install it in your env:  pip install pandas\n"
            f"Import error: {e}"
        )

    df = pd.read_csv(path, engine="python")
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]

    cols = list(df.columns)

    # Detect long-table format:
    # time column patterns
    tcol = _pick_col(cols, [r"^t($|_)", r"^time", r"_s$"])
    # radius column patterns (r, r_a_m, r_c_m, radius, ... )
    rcol = _pick_col(cols, [r"^r($|_)", r"radius", r"_m$"])
    # value column patterns (cs_a_kmol_m3, cs_c_kmol_m3, cs..., concentration...)
    vcol = _pick_col(cols, [r"^cs($|_)", r"cs_", r"concentration", r"conc"])

    if vcol in (tcol, rcol):
        vcol = None

    if tcol and rcol and vcol:
        df[tcol] = pd.to_numeric(df[tcol], errors="coerce")
        df[rcol] = pd.to_numeric(df[rcol], errors="coerce")
        df[vcol] = pd.to_numeric(df[vcol], errors="coerce")
        df = df.dropna(subset=[tcol, rcol, vcol])

        nt = df[tcol].nunique()
        nr = df[rcol].nunique()
        nrow = len(df)

        # If looks like full grid, pivot
        if nt >= 2 and nr >= 2 and abs(nt * nr - nrow) / max(1, nrow) < 0.02:
            piv = df.pivot_table(index=tcol, columns=rcol, values=vcol, aggfunc="mean")
            piv = piv.sort_index().sort_index(axis=1)

            t_vec = piv.index.to_numpy(dtype=float)
            r_vec = piv.columns.to_numpy(dtype=float)
            Z = piv.to_numpy(dtype=float)
            return t_vec, r_vec, Z

    # Fallback: wide format (time + many r columns)
    first = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    if first.notna().mean() > 0.95 and df.shape[1] >= 2:
        t_vec = first.to_numpy(dtype=float)
        r_hdr = pd.to_numeric(df.columns[1:], errors="coerce")
        if r_hdr.notna().all():
            r_vec = r_hdr.to_numpy(dtype=float)
        else:
            r_vec = np.arange(df.shape[1] - 1, dtype=float)
        Z = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        return t_vec, r_vec, Z

    # Last fallback: numpy matrix
    raw = np.genfromtxt(path, delimiter=",", dtype=float)
    if raw.ndim != 2:
        raise ValueError(f"CSV format not recognized: {path}")
    Z = raw
    t_vec = np.arange(Z.shape[0], dtype=float)
    r_vec = np.arange(Z.shape[1], dtype=float)
    return t_vec, r_vec, Z


def sort_axes(t_vec, r_vec, Z):
    t_vec = np.asarray(t_vec, dtype=float)
    r_vec = np.asarray(r_vec, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if Z.shape != (t_vec.size, r_vec.size):
        raise ValueError(f"Shape mismatch: Z{Z.shape} vs t({t_vec.size}) r({r_vec.size})")

    ti = np.argsort(t_vec)
    ri = np.argsort(r_vec)
    t_sorted = t_vec[ti]
    r_sorted = r_vec[ri]
    Z_sorted = Z[np.ix_(ti, ri)]
    return t_sorted, r_sorted, Z_sorted


def upsample_r_linear(t_vec, r_vec, Z, factor: int):
    """
    Upsample along r by linear interpolation.
    factor=2 -> 55 -> 109 (approx)
    """
    factor = int(factor)
    if factor <= 1 or r_vec.size < 2:
        return t_vec, r_vec, Z

    t_vec, r_vec, Z = sort_axes(t_vec, r_vec, Z)

    new_n = (r_vec.size - 1) * factor + 1
    r_new = np.linspace(r_vec[0], r_vec[-1], new_n, dtype=float)

    Z_new = np.empty((Z.shape[0], r_new.size), dtype=float)
    for i in range(Z.shape[0]):
        Z_new[i, :] = np.interp(r_new, r_vec, Z[i, :])

    return t_vec, r_new, Z_new


def auto_r_to_um(r_vec: np.ndarray):
    rmax = float(np.nanmax(r_vec))
    if rmax < 1e-3:  # meter-scale particle radius
        return r_vec * 1e6, "r (μm)"
    return r_vec, "r (μm)"


def make_axis_text_bold(ax):
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.zaxis.label.set_fontweight("bold")
    ax.title.set_fontweight("bold")
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    for lbl in ax.get_zticklabels():
        lbl.set_fontweight("bold")


def plot_surface_3d(
    t_vec, r_vec, Z,
    title: str,
    zlabel: str,
    out_png: str,
    refine_factor: int,
    decimate: int,
    keep_open: bool,
    save_png: bool,
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    cmap = get_nature_cmap()

    t_vec, r_vec, Z = sort_axes(t_vec, r_vec, Z)

    # downsample for interactive
    decimate = max(1, int(decimate))
    if decimate > 1:
        t_use = t_vec[::decimate]
        r_use = r_vec[::decimate]
        Z_use = Z[::decimate, ::decimate]
    else:
        t_use, r_use, Z_use = t_vec, r_vec, Z

    r_plot, xlabel = auto_r_to_um(r_use)
    T, R = np.meshgrid(t_use, r_plot, indexing="ij")

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")

    # plot_surface default downsamples ~50x50, override with rcount/ccount
    refine_factor = max(1, int(refine_factor))
    base_r = min(50, Z_use.shape[0])
    base_c = min(50, Z_use.shape[1])
    rcount = min(Z_use.shape[0], int(base_r * refine_factor))
    ccount = min(Z_use.shape[1], int(base_c * refine_factor))

    surf = ax.plot_surface(
        R, T, Z_use,
        cmap=cmap,
        rcount=rcount,
        ccount=ccount,
        linewidth=0,
        edgecolor="none",
        antialiased=(decimate == 1),
        shade=False,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("t (s)")
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    cbar = fig.colorbar(surf, shrink=0.62, pad=0.08)
    cbar.ax.tick_params(labelsize=12)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight("bold")

    ax.view_init(elev=25, azim=-60)
    make_axis_text_bold(ax)

    # Avoid tight_layout warnings in 3D
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.06, top=0.92)

    if save_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=300, bbox_inches="tight")

    if not keep_open:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cs_a", type=str, default=DEFAULT_CS_A)
    parser.add_argument("--cs_c", type=str, default=DEFAULT_CS_C)
    parser.add_argument("--out_dir", type=str, default="Figures3D_Truth_v2")

    # default: show windows
    parser.add_argument("--no_show", action="store_true", help="save only, do not show windows")

    # patch refinement
    parser.add_argument("--save_refine", type=int, default=2, help="saved PNG refine factor (default 2)")
    parser.add_argument("--show_refine", type=int, default=1, help="interactive show refine factor (default 1)")

    # interactive speed
    parser.add_argument("--decimate_show", type=int, default=2, help="interactive decimate factor (default 2)")

    # key for 55-r data
    parser.add_argument("--interp_r", type=int, default=2, help="r upsample factor (default 2)")

    args = parser.parse_args()

    do_show = (not args.no_show)
    setup_matplotlib(show=do_show)

    # Load and pivot to grid
    t_a, r_a, Z_a = load_truth_surface_csv(args.cs_a)
    t_c, r_c, Z_c = load_truth_surface_csv(args.cs_c)

    # r interpolation (55 -> 109 if interp_r=2)
    t_a, r_a, Z_a = upsample_r_linear(t_a, r_a, Z_a, factor=args.interp_r)
    t_c, r_c, Z_c = upsample_r_linear(t_c, r_c, Z_c, factor=args.interp_r)

    os.makedirs(args.out_dir, exist_ok=True)
    out_png_a = os.path.abspath(os.path.join(args.out_dir, "truth_surf_cs_a_v2.png"))
    out_png_c = os.path.abspath(os.path.join(args.out_dir, "truth_surf_cs_c_v2.png"))

    # Save high-quality PNGs
    plot_surface_3d(
        t_a, r_a, Z_a,
        title="Anode Cs(t,r) truth surface (v2)",
        zlabel="C_s,Li,an (kmol/m^3)",
        out_png=out_png_a,
        refine_factor=args.save_refine,
        decimate=1,
        keep_open=False,
        save_png=True,
    )
    plot_surface_3d(
        t_c, r_c, Z_c,
        title="Cathode Cs(t,r) truth surface (v2)",
        zlabel="C_s,Li,ca (kmol/m^3)",
        out_png=out_png_c,
        refine_factor=args.save_refine,
        decimate=1,
        keep_open=False,
        save_png=True,
    )

    print("INFO: Saved PNGs:")
    print("  ", out_png_a)
    print("  ", out_png_c)
    print("INFO: Shapes (after interp_r):")
    print("  cs_a:", Z_a.shape, "t:", t_a.size, "r:", r_a.size)
    print("  cs_c:", Z_c.shape, "t:", t_c.size, "r:", r_c.size)

    # Show interactive
    if do_show:
        import matplotlib.pyplot as plt
        plot_surface_3d(
            t_a, r_a, Z_a,
            title="Anode Cs(t,r) truth surface (v2, interactive)",
            zlabel="C_s,Li,an (kmol/m^3)",
            out_png=out_png_a,
            refine_factor=args.show_refine,
            decimate=args.decimate_show,
            keep_open=True,
            save_png=False,
        )
        plot_surface_3d(
            t_c, r_c, Z_c,
            title="Cathode Cs(t,r) truth surface (v2, interactive)",
            zlabel="C_s,Li,ca (kmol/m^3)",
            out_png=out_png_c,
            refine_factor=args.show_refine,
            decimate=args.decimate_show,
            keep_open=True,
            save_png=False,
        )
        plt.show()


if __name__ == "__main__":
    main()
