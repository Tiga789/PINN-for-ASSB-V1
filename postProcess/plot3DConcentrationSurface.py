# -*- coding: utf-8 -*-
"""
3D surface plots for SPM PINN predictions:
  - cs_a(t, r): anode solid lithium concentration
  - cs_c(t, r): cathode solid lithium concentration

Run example (PowerShell):
  D:\\Anaconda\\envs\\pinnstripes\\python.exe .\\pinn_spm_param\\postProcess\\plot3DConcentrationSurface.py `
      -mf pinn_spm_param\\ModelFin_1 -p 0.8 3.0 -nosimp -nt 256 -nr 256 -ff Figures3D -v
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf

# ============================================================
# Plot mesh refine factor:
#   Matplotlib plot_surface default downsamples to 50×50.
#   Set factor=2  -> 100×100, i.e., each old patch -> 4 patches.
# ============================================================
PLOT_REFINE_FACTOR = 2


# ------------------------------------------------------------
# IMPORTANT: set matplotlib backend BEFORE importing pyplot
# ------------------------------------------------------------
def _setup_matplotlib_backend(verbose: bool):
    import matplotlib
    if verbose:
        # Prefer Qt for best interactive 3D; fallback to Tk if Qt not available
        try:
            import PyQt5  # noqa: F401
            matplotlib.use("Qt5Agg")
        except Exception:
            matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")  # save only

    # -----------------------------
    # Global font: Times New Roman + bold
    # -----------------------------
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


def _get_nature_cmap():
    """Nature-style cool-to-warm palette from your image."""
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#403990", "#80A6E2", "#FBDD85", "#F46F43", "#CF3D3E"]
    return LinearSegmentedColormap.from_list("nature_palette", colors, N=256)


# ------------------------------------------------------------
# Robust imports from project util/
# ------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
UTIL_DIR = os.path.abspath(os.path.join(THIS_DIR, "../util"))
if UTIL_DIR not in sys.path:
    sys.path.insert(0, UTIL_DIR)

import argument
from init_pinn import initialize_nn_from_params_config, safe_load
from forwardPass import make_var_params_from_data, pinn_pred, from_param_list_to_str
from keras.backend import set_floatx

set_floatx("float64")


def _build_tr_grid(params, params_list, n_t: int, n_r: int, electrode: str):
    """
    Build structured grid in (t, r) for cs prediction.

    electrode:
      - "an": r in [rmin, Rs_a]
      - "ca": r in [rmin, Rs_c]
    """
    tmin = float(params["tmin"])
    tmax = float(params["tmax"])
    rmin = float(params["rmin"])

    if electrode == "an":
        rmax = float(params["Rs_a"])
    else:
        rmax = float(params["Rs_c"])

    t_vec = np.linspace(tmin, tmax, n_t, dtype=np.float64)
    r_vec = np.linspace(rmin, rmax, n_r, dtype=np.float64)

    # Meshgrid (n_t, n_r)
    T, R = np.meshgrid(t_vec, r_vec, indexing="ij")
    x_tr = np.stack([T.reshape(-1), R.reshape(-1)], axis=1)  # (n_t*n_r, 2)

    # Duplicate params for each point
    p = np.array(params_list, dtype=np.float64).reshape(1, -1)
    p_rep = np.repeat(p, x_tr.shape[0], axis=0)

    return t_vec, r_vec, x_tr, p_rep


def _predict_cs_surfaces(nn, params_list, n_t: int, n_r: int):
    """
    Predict cs_a(t,r) and cs_c(t,r) on structured grids.
    Returns:
      t_vec, r_a_vec, r_c_vec, cs_a (n_t,n_r), cs_c (n_t,n_r)
    """
    # Build grids
    t_vec, r_a_vec, x_cs_a, p_cs_a = _build_tr_grid(nn.params, params_list, n_t, n_r, "an")
    _, r_c_vec, x_cs_c, p_cs_c = _build_tr_grid(nn.params, params_list, n_t, n_r, "ca")

    # time-only inputs (keep structure consistent)
    t_only = x_cs_a[:, [0]]

    data_dict = {
        "var_phie": t_only,
        "params_phie": p_cs_a,
        "var_phis_c": t_only,
        "params_phis_c": p_cs_a,
        "var_cs_a": x_cs_a,
        "params_cs_a": p_cs_a,
        "var_cs_c": x_cs_c,
        "params_cs_c": p_cs_c,
    }

    var_dict, params_dict = make_var_params_from_data(nn, data_dict)
    pred = pinn_pred(nn, var_dict, params_dict)

    cs_a = tf.reshape(pred["cs_a"], (n_t, n_r)).numpy()
    cs_c = tf.reshape(pred["cs_c"], (n_t, n_r)).numpy()

    return t_vec, r_a_vec, r_c_vec, cs_a, cs_c


def _make_axis_text_bold(ax):
    """Force bold for 3D axis labels and tick labels."""
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


def _plot_surface(t_vec, r_vec_m, Z, title, zlabel, out_png, keep_open: bool):
    """
    3D surface plot: x=r(um), y=t(s), z=Z.

    Key change:
      - plot_surface default will downsample to 50×50 (rcount/ccount=50)
      - here we refine it by PLOT_REFINE_FACTOR (2 => 100×100),
        which makes each old color patch split into 4.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    cmap = _get_nature_cmap()

    r_um = r_vec_m * 1e6
    T, R = np.meshgrid(t_vec, r_um, indexing="ij")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # ---- refine surface tessellation (display mesh) ----
    # default base is 50; refine by factor=2 -> 100
    base_r = min(50, Z.shape[0])
    base_c = min(50, Z.shape[1])
    rcount = min(Z.shape[0], int(base_r * PLOT_REFINE_FACTOR))
    ccount = min(Z.shape[1], int(base_c * PLOT_REFINE_FACTOR))

    surf = ax.plot_surface(
        R, T, Z,
        cmap=cmap,
        rcount=rcount,
        ccount=ccount,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        shade=False,  # 让颜色更忠实反映数值，减少“分块光照感”
    )

    ax.set_xlabel("r (μm)")
    ax.set_ylabel("t (s)")
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    cbar = fig.colorbar(surf, shrink=0.6, pad=0.08)
    cbar.ax.tick_params(labelsize=12)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight("bold")

    ax.view_init(elev=25, azim=-60)

    _make_axis_text_bold(ax)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)

    if not keep_open:
        plt.close(fig)


def main():
    args = argument.initArg()

    verbose = bool(getattr(args, "verbose", False))
    _setup_matplotlib_backend(verbose)

    # Choose realistic vs simpler SPM
    if getattr(args, "simpleModel", False):
        from spm_simpler import makeParams
        print("INFO: USING SIMPLE SPM MODEL")
    else:
        from spm import makeParams
        print("INFO: USING REALISTIC SPM MODEL")
    params = makeParams()

    # Parse parameter list
    if args.params_list is None or len(args.params_list) != 2:
        raise RuntimeError("ERROR: param list is mandatory. Use: -p <deg_i0_a> <deg_ds_c>")
    params_list = [float(x) for x in args.params_list]

    # Grid size (prediction grid)
    n_t = int(getattr(args, "n_t", 256))
    n_r = int(getattr(args, "n_r", 256))

    # Load model
    modelFolder = getattr(args, "modelFolder", None)
    if modelFolder is None:
        raise RuntimeError("ERROR: model folder is mandatory. Use: -mf <modelFolder>")

    cfg_path = os.path.join(modelFolder, "config.json")
    w_path = os.path.join(modelFolder, "best.weights.h5")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"ERROR: cannot find {cfg_path}")
    if not os.path.exists(w_path):
        raise FileNotFoundError(f"ERROR: cannot find {w_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        configDict = json.load(f)

    nn = initialize_nn_from_params_config(params, configDict)
    nn = safe_load(nn, w_path)

    # Predict surfaces
    t_vec, r_a_vec, r_c_vec, cs_a, cs_c = _predict_cs_surfaces(nn, params_list, n_t, n_r)

    # Output folder
    fig_root = getattr(args, "figureFolder", None) or "Figures3D"
    model_tag = os.path.normpath(modelFolder).lstrip(".\\/").strip("\\/").replace("\\", "_").replace("/", "_")
    ptag = from_param_list_to_str(params_list).replace(".", "p")

    out_dir = os.path.join(fig_root, model_tag)
    out_png_a = os.path.join(out_dir, f"surf_cs_a_{ptag}.png")
    out_png_c = os.path.join(out_dir, f"surf_cs_c_{ptag}.png")

    _plot_surface(
        t_vec, r_a_vec, cs_a,
        title="Anode Cs(t,r) surface",
        zlabel="C_s,Li,an (kmol/m^3)",
        out_png=out_png_a,
        keep_open=verbose,
    )

    _plot_surface(
        t_vec, r_c_vec, cs_c,
        title="Cathode Cs(t,r) surface",
        zlabel="C_s,Li,ca (kmol/m^3)",
        out_png=out_png_c,
        keep_open=verbose,
    )

    print(f"INFO: Saved 3D surfaces to: {out_dir}")
    print(f"      {out_png_a}")
    print(f"      {out_png_c}")
    print(f"INFO: plot_surface mesh refined by factor = {PLOT_REFINE_FACTOR} "
          f"(default 50x50 -> {min(50, n_t) * PLOT_REFINE_FACTOR}x{min(50, n_r) * PLOT_REFINE_FACTOR}, capped by grid)")

    if verbose:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
