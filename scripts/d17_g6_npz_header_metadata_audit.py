import argparse
import csv
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npfmt


PARAM_PATTERNS = [
    "capacity", "capacity_scale", "qeff", "q_eff",
    "theta_positive", "theta_negative", "theta_a_initial", "theta_c_initial",
    "theta_a0", "theta_c0", "initial",
    "phie_ohmic", "ohmic_scale", "phie_source", "phis_c_source",
    "csmax", "cs_max",
    "alpha_d", "alpha_j",
    "particle", "radius", "r_particle",
    "diffusion", "ds", "d_eff",
    "current_sign", "sign_convention",
    "source", "replay", "config", "script", "hash", "sha", "stage",
]

STATE_KEYS = {
    "cs_a", "cs_c", "theta_a", "theta_c", "phie", "phis_c",
    "cs_a_soft", "cs_c_soft", "theta_a_soft", "theta_c_soft",
    "phie_soft", "phis_c_soft",
}

PROFILE_VECTOR_HINTS = [
    "t_global_s", "cycle_id", "step_id", "step_type",
    "I", "current", "voltage", "temperature",
    "cbar", "J_", "j_", "theta_a_mean", "theta_c_mean",
]


def npy_header_from_npz_member(zf: zipfile.ZipFile, member: str):
    with zf.open(member, "r") as fp:
        version = npfmt.read_magic(fp)
        try:
            shape, fortran_order, dtype = npfmt._read_array_header(
                fp, version, max_header_size=2**20
            )
        except TypeError:
            shape, fortran_order, dtype = npfmt._read_array_header(fp, version)
    return tuple(shape), bool(fortran_order), str(dtype)


def nelem(shape):
    if shape == ():
        return 1
    n = 1
    for x in shape:
        try:
            n *= int(x)
        except Exception:
            return None
    return n


def safe_value_repr(arr, max_chars=500):
    try:
        if getattr(arr, "shape", None) == ():
            val = arr.item()
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
            return repr(val)[:max_chars]
        val = arr.tolist()
        return repr(val)[:max_chars]
    except Exception as e:
        return f"<value_repr_failed: {type(e).__name__}: {e}>"


def classify_key(key, n, dtype):
    lk = key.lower()
    if key in STATE_KEYS:
        return "state_target_array"
    if any(p in lk for p in PARAM_PATTERNS):
        return "candidate_generation_metadata"
    if any(h.lower() in lk for h in PROFILE_VECTOR_HINTS):
        return "profile_time_or_inventory_vector"
    if n is not None and n <= 256:
        return "small_scalar_or_metadata"
    if "r_grid" in lk or lk in {"r", "r_a", "r_c", "weights", "volume_weights"}:
        return "radial_grid_or_weights"
    return "large_or_other_array"


def inspect_one(npz_path: Path, summary_path: Path, small_max_elements: int):
    out = {
        "npz_path": str(npz_path),
        "npz_exists": npz_path.exists(),
        "summary_path": str(summary_path),
        "summary_exists": summary_path.exists(),
        "file_size_gb": None,
        "summary_json": None,
        "keys": [],
        "candidate_metadata_keys": [],
        "state_target_keys": [],
        "small_metadata_values": {},
        "generation_parameter_presence": {},
        "conclusion": {},
    }

    if npz_path.exists():
        out["file_size_gb"] = npz_path.stat().st_size / (1024**3)

    if summary_path.exists():
        try:
            out["summary_json"] = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as e:
            out["summary_json"] = {"read_error": f"{type(e).__name__}: {e}"}

    if not npz_path.exists():
        out["conclusion"] = {
            "status": "FAIL",
            "reason": "npz does not exist",
        }
        return out

    with zipfile.ZipFile(npz_path, "r") as zf:
        members = [m for m in zf.namelist() if m.endswith(".npy")]
        header_rows = []
        for m in members:
            key = Path(m).stem
            try:
                shape, fortran_order, dtype = npy_header_from_npz_member(zf, m)
                n = nelem(shape)
                cat = classify_key(key, n, dtype)
                row = {
                    "key": key,
                    "member": m,
                    "shape": list(shape),
                    "dtype": dtype,
                    "fortran_order": fortran_order,
                    "n_elements": n,
                    "category": cat,
                    "small_value_repr": "",
                }
                header_rows.append(row)
            except Exception as e:
                header_rows.append({
                    "key": key,
                    "member": m,
                    "shape": None,
                    "dtype": None,
                    "fortran_order": None,
                    "n_elements": None,
                    "category": "header_read_error",
                    "small_value_repr": f"{type(e).__name__}: {e}",
                })

    # Load only small arrays / metadata candidates. This does not load large state arrays.
    try:
        data = np.load(npz_path, allow_pickle=True)
        try:
            for row in header_rows:
                key = row["key"]
                n = row["n_elements"]
                cat = row["category"]
                should_load = (
                    n is not None
                    and n <= small_max_elements
                    and cat in {
                        "candidate_generation_metadata",
                        "small_scalar_or_metadata",
                        "radial_grid_or_weights",
                    }
                )
                if should_load and key in data.files:
                    try:
                        arr = data[key]
                        row["small_value_repr"] = safe_value_repr(arr)
                        out["small_metadata_values"][key] = row["small_value_repr"]
                    except Exception as e:
                        row["small_value_repr"] = f"<load_failed: {type(e).__name__}: {e}>"
        finally:
            data.close()
    except Exception as e:
        out["small_metadata_values"]["np_load_error"] = f"{type(e).__name__}: {e}"

    out["keys"] = header_rows
    out["candidate_metadata_keys"] = [
        r["key"] for r in header_rows if r["category"] == "candidate_generation_metadata"
    ]
    out["state_target_keys"] = [
        r["key"] for r in header_rows if r["category"] == "state_target_array"
    ]

    all_keys_lower = " ".join([r["key"].lower() for r in header_rows])
    small_values_lower = " ".join([str(v).lower() for v in out["small_metadata_values"].values()])
    combined = all_keys_lower + " " + small_values_lower + " " + json.dumps(out.get("summary_json") or {}).lower()

    checks = {
        "capacity_scale_found": any(s in combined for s in ["capacity_scale", "capacity scale", "capacity_ah", "capacity"]),
        "theta_positive_initial_found": any(s in combined for s in ["theta_positive_initial", "theta_c_initial", "theta_c0"]),
        "theta_negative_initial_found": any(s in combined for s in ["theta_negative_initial", "theta_a_initial", "theta_a0"]),
        "phie_ohmic_scale_found": any(s in combined for s in ["phie_ohmic", "ohmic_scale", "lumped_current_ohmic"]),
        "source_replay_found": "source_replay" in combined or "solution_replay_profile" in combined,
        "script_or_config_hash_found": any(s in combined for s in ["script_hash", "config_hash", "sha256", "git", "commit"]),
        "cbar_keys_found": any(r["key"].lower().startswith("cbar") or "cbar" in r["key"].lower() for r in header_rows),
        "j_flux_keys_found": any(r["key"].lower().startswith("j_") or "flux" in r["key"].lower() for r in header_rows),
        "theta_mean_keys_found": any("theta_a_mean" in r["key"].lower() or "theta_c_mean" in r["key"].lower() for r in header_rows),
    }
    out["generation_parameter_presence"] = checks

    deployable_exact_replay_params = (
        checks["capacity_scale_found"]
        and checks["theta_positive_initial_found"]
        and checks["theta_negative_initial_found"]
        and checks["source_replay_found"]
    )

    out["conclusion"] = {
        "status": "PASS",
        "large_arrays_loaded": False,
        "deployable_exact_replay_params_found": deployable_exact_replay_params,
        "has_script_or_config_hash": checks["script_or_config_hash_found"],
        "recommended_next": (
            "exact_replay_possible_from_npz_metadata"
            if deployable_exact_replay_params
            else "npz_metadata_insufficient_for_exact_replay"
        ),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--softlabel_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--small_max_elements", type=int, default=256)
    args = ap.parse_args()

    softlabel_root = Path(args.softlabel_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    flat_rows = []

    t0 = time.time()
    for cell in args.cells:
        prof_dir = softlabel_root / "profiles" / cell
        npz_path = prof_dir / "solution_softlabels.npz"
        summary_path = prof_dir / "soft_label_summary.json"

        res = inspect_one(npz_path, summary_path, args.small_max_elements)
        res["cell_id"] = cell
        all_results.append(res)

        for row in res.get("keys", []):
            flat_rows.append({
                "cell_id": cell,
                "key": row.get("key"),
                "shape": json.dumps(row.get("shape"), ensure_ascii=False),
                "dtype": row.get("dtype"),
                "n_elements": row.get("n_elements"),
                "category": row.get("category"),
                "small_value_repr": row.get("small_value_repr", ""),
            })

    summary = {
        "protocol": "D17-G6_NPZ_HEADER_SCALAR_METADATA_AUDIT",
        "status": "PASS",
        "training_performed": False,
        "large_arrays_loaded": False,
        "elapsed_s": time.time() - t0,
        "softlabel_root": str(softlabel_root),
        "cells": args.cells,
        "cell_results": all_results,
        "overall": {
            "all_npz_exist": all(r.get("npz_exists") for r in all_results),
            "any_deployable_exact_replay_params_found": any(
                r.get("conclusion", {}).get("deployable_exact_replay_params_found")
                for r in all_results
            ),
            "all_deployable_exact_replay_params_found": all(
                r.get("conclusion", {}).get("deployable_exact_replay_params_found")
                for r in all_results
            ),
            "all_have_script_or_config_hash": all(
                r.get("conclusion", {}).get("has_script_or_config_hash")
                for r in all_results
            ),
        },
    }

    if not summary["overall"]["all_npz_exist"]:
        summary["status"] = "FAIL"
        summary["recommendation"] = "STOP_MISSING_NPZ"
    elif summary["overall"]["all_deployable_exact_replay_params_found"]:
        summary["recommendation"] = "NPZ_METADATA_MAY_SUPPORT_EXACT_REPLAY"
    else:
        summary["recommendation"] = "NPZ_METADATA_INSUFFICIENT_TREAT_ALL55_FINAL_AS_TEACHER_DATASET"

    json_path = out_dir / "D17_G6_NPZ_HEADER_METADATA_AUDIT.json"
    csv_path = out_dir / "D17_G6_NPZ_HEADER_KEYS.csv"

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["cell_id", "key", "shape", "dtype", "n_elements", "category", "small_value_repr"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in flat_rows:
            w.writerow(r)

    print(json.dumps({
        "status": summary["status"],
        "recommendation": summary["recommendation"],
        "out_dir": str(out_dir),
        "summary_json": str(json_path),
        "keys_csv": str(csv_path),
        "overall": summary["overall"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
