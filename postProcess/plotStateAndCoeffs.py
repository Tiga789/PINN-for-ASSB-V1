"""
Plot PINN-predicted state variables AND derived transport/kinetic coefficients.

Outputs Cartesian (x-y) plots for:
  1) cs_a(t,r)  : anode solid lithium concentration
  2) cs_c(t,r)  : cathode solid lithium concentration
  3) phie(t)    : electrolyte potential
  4) phis_c(t)  : cathode solid potential (cell voltage when phis_a=0)
  5) i0_a(t)    : anode exchange current density (computed at particle surface)
  6) Ds_c(t,r)  : cathode solid diffusion coefficient (computed from cs_c)

Example:
  python plotStateAndCoeffs.py -mf ../Model -p 0.5 1.0 -nosimp
Optional:
  -df ../integration_spm   # overlay PDE solution if available
  -nr 128 -nt 256          # control plotting resolution
"""

import json
import os
import sys
import numpy as np

# Make imports robust to the current working directory.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
UTIL_DIR = os.path.abspath(os.path.join(THIS_DIR, "../util"))
if UTIL_DIR not in sys.path:
    sys.path.append(UTIL_DIR)

import argument
import tensorflow as tf
from forwardPass import from_param_list_to_str, make_data_dict_struct
from init_pinn import initialize_nn_from_params_config, safe_load
from keras.backend import set_floatx
from plotsUtil_batt import line_cs_results, line_phi_results
from prettyPlot.plotting import plt, pretty_labels, pretty_legend

set_floatx("float64")


def _build_struct_grid(params, params_list, n_t=64, n_r=64):
    """Flexible version of forwardPass.pinn_pred_struct with user-controlled n_t/n_r."""
    tmin = params["tmin"]
    tmax = params["tmax"]
    rmin = params["rmin"]
    rmax_a = params["Rs_a"]
    rmax_c = params["Rs_c"]

    t_test = np.linspace(tmin, tmax, n_t).reshape(n_t, 1, 1, 1)
    r_test_a = np.linspace(rmin, rmax_a, n_r).reshape(1, n_r, 1, 1)
    r_test_c = np.linspace(rmin, rmax_c, n_r).reshape(1, n_r, 1, 1)

    # Flattened grids for NN evaluation
    t_a = np.repeat(t_test, n_r, axis=1).reshape(-1, 1).astype("float64")
    r_a = np.repeat(r_test_a, n_t, axis=0).reshape(-1, 1).astype("float64")
    t_c = np.repeat(t_test, n_r, axis=1).reshape(-1, 1).astype("float64")
    r_c = np.repeat(r_test_c, n_t, axis=0).reshape(-1, 1).astype("float64")

    # Parameters: one value for every (t,r) point
    params_unr = np.tile(np.array(params_list, dtype=np.float64), (t_a.shape[0], 1))
    return t_test, r_test_a, r_test_c, t_a, r_a, t_c, r_c, params_unr


def _pinn_predict_on_grid(nn, params_list, n_t=64, n_r=64):
    """Run PINN prediction on a structured grid."""
    (
        t_test,
        r_test_a,
        r_test_c,
        t_a,
        r_a,
        t_c,
        r_c,
        params_unr,
    ) = _build_struct_grid(nn.params, params_list, n_t=n_t, n_r=n_r)

    data_dict = {
        "var_phie": t_a,
        "params_phie": params_unr,
        "var_phis_c": t_c,
        "params_phis_c": params_unr,
        "var_cs_a": np.hstack([t_a, r_a]),
        "params_cs_a": params_unr,
        "var_cs_c": np.hstack([t_c, r_c]),
        "params_cs_c": params_unr,
    }

    from forwardPass import make_var_params_from_data, pinn_pred
    var_dict, params_dict = make_var_params_from_data(nn, data_dict)
    pred_dict = pinn_pred(nn, var_dict, params_dict)

    phie = tf.reshape(pred_dict["phie"], (n_t, n_r, 1, 1)).numpy()
    phis_c = tf.reshape(pred_dict["phis_c"], (n_t, n_r, 1, 1)).numpy()
    cs_a = tf.reshape(pred_dict["cs_a"], (n_t, n_r, 1, 1)).numpy()
    cs_c = tf.reshape(pred_dict["cs_c"], (n_t, n_r, 1, 1)).numpy()

    return {
        "phie": phie,
        "phis_c": phis_c,
        "cs_a": cs_a,
        "cs_c": cs_c,
        "t_test": t_test,
        "r_test_a": r_test_a,
        "r_test_c": r_test_c,
    }


def _compute_i0a_surface(params, cs_a_surf, deg_i0_a):
    """Compute i0_a at the anode particle surface (r=Rs_a)."""
    cs_a_surf = np.asarray(cs_a_surf, dtype=np.float64)
    ce = np.ones_like(cs_a_surf) * np.float64(params["ce0"])
    deg_vec = np.ones_like(cs_a_surf) * np.float64(deg_i0_a)

    i0 = params["i0_a"](
        tf.convert_to_tensor(cs_a_surf, tf.float64),
        tf.convert_to_tensor(ce, tf.float64),
        tf.convert_to_tensor(np.float64(params["T"]), tf.float64),
        tf.convert_to_tensor(np.float64(params["alpha_a"]), tf.float64),
        tf.convert_to_tensor(np.float64(params["csanmax"]), tf.float64),
        tf.convert_to_tensor(np.float64(params["R"]), tf.float64),
        tf.convert_to_tensor(deg_vec, tf.float64),
    )
    return i0.numpy()


def _compute_dsc_from_csc(params, cs_c, deg_ds_c):
    """Compute Ds_c from cs_c (can be surface or whole radial profile)."""
    cs_c = np.asarray(cs_c, dtype=np.float64)
    deg_vec = np.ones_like(cs_c) * np.float64(deg_ds_c)
    ds = params["D_s_c"](
        tf.convert_to_tensor(cs_c, tf.float64),
        tf.convert_to_tensor(np.float64(params["T"]), tf.float64),
        tf.convert_to_tensor(np.float64(params["R"]), tf.float64),
        tf.convert_to_tensor(np.float64(params["cscamax"]), tf.float64),
        tf.convert_to_tensor(deg_vec, tf.float64),
    )
    return ds.numpy()


def plot_state_and_coeffs(args):
    # 1) Choose SPM parameter set
    if args.simpleModel:
        from spm_simpler import makeParams
    else:
        from spm import makeParams
    params = makeParams()

    # 2) Parse parameter list
    if args.params_list is None:
        sys.exit("ERROR: param list is mandatory. Use: -p <deg_i0_a> <deg_ds_c>")
    params_list = [float(entry) for entry in args.params_list]
    if len(params_list) != 2:
        sys.exit("ERROR: -p expects exactly two values: deg_i0_a deg_ds_c")
    deg_i0_a, deg_ds_c = params_list

    # 3) Matplotlib backend
    if not args.verbose:
        import matplotlib
        matplotlib.use("Agg")

    modelFolder = args.modelFolder
    dataFolder = args.dataFolder
    figureFolder = args.figureFolder or "Figures"

    # 4) Load NN from config + weights
    if not os.path.exists(os.path.join(modelFolder, "config.json")):
        sys.exit(f"ERROR: cannot find {os.path.join(modelFolder, 'config.json')}")
    with open(os.path.join(modelFolder, "config.json")) as f:
        configDict = json.load(f)

    nn = initialize_nn_from_params_config(params, configDict)
    nn = safe_load(nn, os.path.join(modelFolder, "best.weights.h5"))

    # 5) Predict on grid
    sol = _pinn_predict_on_grid(nn, params_list, n_t=args.n_t, n_r=args.n_r)
    phie = sol["phie"]
    phis_c = sol["phis_c"]
    cs_a = sol["cs_a"]
    cs_c = sol["cs_c"]
    t_test = sol["t_test"]
    r_test_a = sol["r_test_a"]
    r_test_c = sol["r_test_c"]

    t = t_test[:, 0, 0, 0]
    r_a = r_test_a[0, :, 0, 0]
    r_c = r_test_c[0, :, 0, 0]

    # 6) Surface concentrations
    cs_a_surf = cs_a[:, -1, 0, 0]
    cs_c_surf = cs_c[:, -1, 0, 0]

    # 7) Derived quantities
    i0_a_surf = _compute_i0a_surface(params, cs_a_surf, deg_i0_a)
    ds_c_surf = _compute_dsc_from_csc(params, cs_c_surf, deg_ds_c)

    # Optional: PDE overlay
    data_dict = None
    if dataFolder is not None:
        try:
            data_dict = make_data_dict_struct(dataFolder, params_list)
        except Exception:
            data_dict = None

    # 8) Output folder
    modelFolderFig = (
        modelFolder.replace("../", "")
        .replace("/Log", "")
        .replace("/Model", "")
        .replace("Log", "")
        .replace("Model", "")
        .replace("/", "_")
        .replace(".", "")
    )
    os.makedirs(figureFolder, exist_ok=True)
    os.makedirs(os.path.join(figureFolder, modelFolderFig), exist_ok=True)
    string_params = from_param_list_to_str(params_list)

    # 9) Time stamps for radial profiles (you can edit this list)
    time_stamps = [0.0, 200.0, 400.0, float(nn.params["tmax"])]
    time_stamps = [s for s in time_stamps if s <= float(nn.params["tmax"])]

    # (1) cs_a(r) at stamps
    file_path = os.path.join(
        figureFolder, modelFolderFig, f"line_cs_a{string_params}_withCoeff.png"
    )
    if data_dict is not None:
        temp_dat, spac_dat, field_dat = data_dict["cs_a_t"], data_dict["cs_a_r"], data_dict["cs_a"]
    else:
        temp_dat = spac_dat = field_dat = None
    line_cs_results(
        temp_pred=t,
        spac_pred=r_a,
        field_pred=cs_a[:, :, 0, 0],
        time_stamps=time_stamps,
        xlabel=r"r ($\mu$m)",
        ylabel=r"C$_{s,Li,an}$ (kmol/m$^3$)",
        file_path_name=file_path,
        temp_dat=temp_dat,
        spac_dat=spac_dat,
        field_dat=field_dat,
        verbose=args.verbose,
    )

    # (2) cs_c(r) at stamps
    file_path = os.path.join(
        figureFolder, modelFolderFig, f"line_cs_c{string_params}_withCoeff.png"
    )
    if data_dict is not None:
        temp_dat, spac_dat, field_dat = data_dict["cs_c_t"], data_dict["cs_c_r"], data_dict["cs_c"]
    else:
        temp_dat = spac_dat = field_dat = None
    line_cs_results(
        temp_pred=t,
        spac_pred=r_c,
        field_pred=cs_c[:, :, 0, 0],
        time_stamps=time_stamps,
        xlabel=r"r ($\mu$m)",
        ylabel=r"C$_{s,Li,ca}$ (kmol/m$^3$)",
        file_path_name=file_path,
        temp_dat=temp_dat,
        spac_dat=spac_dat,
        field_dat=field_dat,
        verbose=args.verbose,
    )

    # (3) phie(t) & phis_c(t)
    file_path = os.path.join(
        figureFolder, modelFolderFig, f"line_phi{string_params}_withCoeff.png"
    )
    if data_dict is not None:
        temp_dat = data_dict["phie_t"]
        phie_dat = data_dict["phie"]
        phis_c_dat = data_dict["phis_c"]
    else:
        temp_dat = phie_dat = phis_c_dat = None
    line_phi_results(
        temp_pred=t,
        field_phie_pred=phie[:, 0, 0, 0],
        field_phis_c_pred=phis_c[:, 0, 0, 0],
        file_path_name=file_path,
        temp_dat=temp_dat,
        field_phie_dat=phie_dat,
        field_phis_c_dat=phis_c_dat,
        verbose=args.verbose,
    )

    # (4) i0_a(t) at surface
    fig = plt.figure()
    plt.plot(t, i0_a_surf, linewidth=3, label=r"$i_{0,an}$ (surface)")
    pretty_legend()
    pretty_labels("time (s)", r"$i_{0,an}$", 14, title="Anode exchange current")
    if not args.verbose:
        plt.savefig(os.path.join(figureFolder, modelFolderFig, f"line_i0_a{string_params}.png"))
        plt.close()

    # (5) Ds_c(t) at surface
    fig = plt.figure()
    plt.plot(t, ds_c_surf, linewidth=3, label=r"$D_{s,ca}$ (surface)")
    pretty_legend()
    pretty_labels("time (s)", r"$D_{s,ca}$", 14, title="Cathode solid diffusion (surface)")
    if not args.verbose:
        plt.savefig(os.path.join(figureFolder, modelFolderFig, f"line_Ds_c{string_params}.png"))
        plt.close()

    # (6) Ds_c(r) at stamps
    fig = plt.figure()
    for stamp in time_stamps:
        it = int(np.argmin(np.abs(t - stamp)))
        ds_profile = _compute_dsc_from_csc(params, cs_c[it, :, 0, 0], deg_ds_c)
        plt.plot(r_c * 1e6, ds_profile, linewidth=3, label=f"t={t[it]:.0f}s")
    pretty_legend()
    pretty_labels(r"r ($\mu$m)", r"$D_{s,ca}$", 14, title="Cathode Ds(r) at time stamps")
    if not args.verbose:
        plt.savefig(os.path.join(figureFolder, modelFolderFig, f"line_Ds_c_r{string_params}.png"))
        plt.close()

    if args.verbose:
        plt.show()

    print(f"INFO: Saved figures to {os.path.join(figureFolder, modelFolderFig)}")


if __name__ == "__main__":
    args = argument.initArg()
    plot_state_and_coeffs(args)
