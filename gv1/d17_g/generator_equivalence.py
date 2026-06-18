from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
    from numpy.lib import format as np_format
except Exception:  # pragma: no cover
    np = None
    np_format = None

GENERATOR_FILES_DEFAULT: List[str] = [
    "scripts/d15_p0_generate_p2dlite_rg_softlabels.py",
    "scripts/d15_p3c_generate_batch2_15cell_rg_softlabels.py",
    "scripts/d15_p3c_generate_batch2_rg_softlabels.py",
    "scripts/d15_p4b_generate_ready18_rg_softlabels.py",
    "scripts/d15_p4d_full_generate_one_rg_softlabel.py",
    "scripts/d15_p4d_generate_one_smoke_profile.py",
    "gv1/p2dlite_rg/radial_solver.py",
    "gv1/p2dlite_rg/io_utils.py",
    "gv1/p2dlite_rg/data.py",
    "gv1/p2dlite_rg/model.py",
    "gv1/p2dlite_rg/train_eval.py",
]

REQUIRED_STATE_KEYS: List[str] = ["cs_a", "cs_c", "theta_a", "theta_c", "phie", "phis_c"]
RG_KEYS: List[str] = [
    "cbar_a", "cbar_c", "cs_a_surface", "cs_c_surface", "cs_a_center", "cs_c_center",
    "grad_a_surface_center", "grad_c_surface_center", "grad_a_surface_mean", "grad_c_surface_mean",
    "J_a_eff_rg", "J_c_eff_rg", "D_a_eff_rg", "D_c_eff_rg",
    "r_a", "r_c", "radial_volume_weights_a", "radial_volume_weights_c",
]

# Scalar/string keys can be read safely from .npz without loading the 52GB state arrays.
SAFE_SCALAR_KEYS: List[str] = [
    "radial_solver_version",
    "radial_gradient_quality_flag",
    "source_profile_npz",
    "source_p2dlite_v1_key_a",
    "source_p2dlite_v1_key_c",
    "source_flux_method_a",
    "source_flux_method_c",
    "phis_c_voltage_preserved_from_source",
    "source_file",
    "cell_uid",
    "batch",
    "protocol",
]

PATTERN_CATALOG: Dict[str, List[str]] = {
    "rg_generate_from_source": ["generate_rg_profile", "cs_a_source_p2dlite_v1", "cs_c_source_p2dlite_v1"],
    "cbar_source_priority": ["get_cbar_field", "weighted_mean_from_source_cs", "source_cbar_field"],
    "flux_source_priority": ["get_j_field", "source_j_field", "infer_surface_flux_from_cbar", "inferred_J_from_cbar_derivative"],
    "preserve_voltage_phi": ["Preserve voltage and phi labels", "preserve_source_voltage_labels", "preserve_source_phi_labels", "phis_c_voltage_preserved_from_source"],
    "p4d_voltage_as_phis": ["phis_c_soft = V", "phis_c = V", "voltage_exp", "out['phis_c']"],
    "p4d_phie_ohmic": ["phie_ohmic_scale", "phie_ohmic_scale_V_per_A", "phie =", "out['phie']"],
    "fixed_theta_initial": ["theta_positive_initial", "theta_negative_initial", "theta_c_mean", "theta_a_mean"],
    "capacity_current_integral": ["capacity_scale_Ah", "cum", "coulomb", "current_integral", "np.cumsum"],
    "radial_fvm_core": ["backward", "implicit", "finite", "volume", "zero", "cbar", "generate_rg_profile"],
    "theta_bounds_clip": ["theta_min_clip", "theta_max_clip", "gradient_clip_normalized"],
    "train_supervised_targets": ["build_targets", "theta_a", "theta_c", "phie", "phis_c"],
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_load(path: Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except Exception as exc:
        return {"_read_error": repr(exc)} if default is None else default


def json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block_size), b""):
            h.update(b)
    return h.hexdigest()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def line_contexts(text: str, needles: Sequence[str], max_hits: int = 8) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        for n in needles:
            if n and n in line:
                out.append({"line": i, "needle": n, "text": line.strip()[:260]})
                if len(out) >= max_hits:
                    return out
    return out


def scan_generator_code(project_root: Path, generator_files: Sequence[str]) -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    aggregate = Counter()
    missing: List[str] = []
    for rel in generator_files:
        p = project_root / rel
        rec: Dict[str, Any] = {"relative_path": rel, "exists": bool(p.exists())}
        if not p.exists():
            missing.append(rel)
            files.append(rec)
            continue
        txt = read_text(p)
        rec.update({"size_bytes": p.stat().st_size, "sha256": sha256_file(p), "line_count": len(txt.splitlines()), "pattern_hits": {}})
        for group, needles in PATTERN_CATALOG.items():
            count = sum(txt.count(n) for n in needles)
            present = count > 0
            rec["pattern_hits"][group] = {
                "present": present,
                "count": count,
                "contexts": line_contexts(txt, needles, max_hits=6) if present else [],
            }
            if present:
                aggregate[group] += 1
        files.append(rec)
    essential = ["rg_generate_from_source", "cbar_source_priority", "flux_source_priority", "radial_fvm_core", "theta_bounds_clip"]
    missing_groups = [g for g in essential if aggregate.get(g, 0) == 0]
    # D15-P4D markers are important for ALL55, but old/local branches may be compressed/minified; REVIEW not FAIL.
    review_groups = [g for g in ["p4d_voltage_as_phis", "p4d_phie_ohmic", "fixed_theta_initial", "capacity_current_integral"] if aggregate.get(g, 0) == 0]
    status = "PASS" if not missing_groups and len(missing) <= 2 else "REVIEW"
    return {
        "generator_files_requested": list(generator_files),
        "files": files,
        "missing_files": missing,
        "pattern_group_file_counts": dict(aggregate),
        "missing_essential_pattern_groups": missing_groups,
        "missing_review_pattern_groups": review_groups,
        "status": status,
    }


def read_npz_header_shapes(npz_path: Path) -> Tuple[Dict[str, Any], Optional[str]]:
    if np_format is None:
        return {}, "numpy is not importable; cannot parse npz headers"
    result: Dict[str, Any] = {}
    try:
        with zipfile.ZipFile(npz_path, "r") as zf:
            for info in zf.infolist():
                if not info.filename.endswith(".npy"):
                    continue
                key = info.filename[:-4]
                try:
                    with zf.open(info, "r") as f:
                        version = np_format.read_magic(f)
                        if version == (1, 0):
                            shape, fortran_order, dtype = np_format.read_array_header_1_0(f)
                        elif version in {(2, 0), (3, 0)}:
                            shape, fortran_order, dtype = np_format.read_array_header_2_0(f)
                        else:
                            raise ValueError(f"unsupported npy version {version}")
                    result[key] = {
                        "shape": list(shape) if isinstance(shape, tuple) else [],
                        "dtype": str(dtype),
                        "fortran_order": bool(fortran_order),
                        "compressed_size": int(info.compress_size),
                        "file_size": int(info.file_size),
                    }
                except Exception as exc:
                    result[key] = {"header_error": repr(exc), "compressed_size": int(info.compress_size), "file_size": int(info.file_size)}
        return result, None
    except Exception as exc:
        return {}, repr(exc)


def _scalar_to_py(x: Any) -> Any:
    try:
        if hasattr(x, "shape") and getattr(x, "shape", None) == ():
            x = x.item()
        elif hasattr(x, "size") and x.size == 1:
            x = x.reshape(-1)[0].item()
    except Exception:
        pass
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, (str, bool, int, float)):
        return x
    return str(x)


def read_npz_safe_scalars(npz_path: Path, available_keys: Iterable[str]) -> Dict[str, Any]:
    """Read only small scalar/string metadata arrays from a soft-label NPZ.

    This intentionally avoids reading large cs/theta arrays.
    """
    if np is None:
        return {}
    keys = set(available_keys)
    wanted = [k for k in SAFE_SCALAR_KEYS if k in keys]
    out: Dict[str, Any] = {}
    if not wanted:
        return out
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            for k in wanted:
                try:
                    arr = data[k]
                    # Accept scalar/string/boolean/small metadata only.
                    if arr.shape == () or arr.size <= 4 or arr.dtype.kind in {"U", "S", "b"}:
                        out[k] = _scalar_to_py(arr)
                except Exception as exc:
                    out[k] = f"<READ_ERROR {exc!r}>"
    except Exception as exc:
        out["_scalar_read_error"] = repr(exc)
    return out


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def canonical_id_for_record(record: Mapping[str, Any]) -> str:
    return str(record.get("canonical_cell_uid") or record.get("cell_uid") or record.get("profile_id") or "UNKNOWN")


def resolve_npz_path(record: Mapping[str, Any], softlabel_root: Optional[Path]) -> Optional[Path]:
    p = record.get("softlabel_npz")
    if p:
        pp = Path(str(p))
        if pp.exists():
            return pp
    pdir = record.get("softlabel_dir")
    if pdir:
        pp = Path(str(pdir)) / "solution_softlabels.npz"
        if pp.exists():
            return pp
    if softlabel_root is not None:
        candidates: List[Path] = []
        for k in ["cell_uid", "canonical_cell_uid", "profile_id"]:
            v = record.get(k)
            if not v:
                continue
            s = str(v).replace("\\", "/").strip("/")
            s_base = s.split("/")[-1]
            candidates.append(softlabel_root / "profiles" / s / "solution_softlabels.npz")
            candidates.append(softlabel_root / "profiles" / s_base / "solution_softlabels.npz")
            m = re.match(r"(Batch-\d+)_([^_]+)_battery-(\d+)$", s_base)
            if m:
                candidates.append(softlabel_root / "profiles" / f"{m.group(1)}_battery-{m.group(3)}" / "solution_softlabels.npz")
        for c in candidates:
            if c.exists():
                return c
    return None


def find_sidecar_summary(record: Mapping[str, Any], softlabel_npz: Path) -> Optional[Path]:
    p = record.get("softlabel_summary")
    if p:
        pp = Path(str(p))
        if pp.exists():
            return pp
    for name in ["soft_label_summary.json", "summary.json", "D15_P2DLITE_RG_PROFILE_SUMMARY.json"]:
        cand = softlabel_npz.with_name(name)
        if cand.exists():
            return cand
    return None


def infer_from_keys(keys: Iterable[str]) -> Dict[str, Any]:
    keyset = set(keys)
    return {
        "has_required_state_keys": all(k in keyset for k in REQUIRED_STATE_KEYS),
        "missing_required_state_keys": [k for k in REQUIRED_STATE_KEYS if k not in keyset],
        "has_rg_diagnostics": any(k in keyset for k in RG_KEYS),
        "rg_keys_present": [k for k in RG_KEYS if k in keyset],
        "has_source_v1_states": any(k in keyset for k in ["cs_a_source_p2dlite_v1", "cs_c_source_p2dlite_v1"]),
        "has_source_flux_method_keys": any(k in keyset for k in ["source_flux_method_a", "source_flux_method_c"]),
        "has_radial_solver_version": "radial_solver_version" in keyset,
        "has_preserved_voltage_flag": "phis_c_voltage_preserved_from_source" in keyset,
        "has_replay_observed_keys": any(k in keyset for k in ["voltage_exp", "I_profile", "temperature_C", "t_global_s"]),
    }


def shape_str(headers: Mapping[str, Any], key: str) -> str:
    v = headers.get(key)
    if not isinstance(v, Mapping):
        return ""
    if "shape" in v:
        return f"{v.get('shape')} {v.get('dtype', '')}".strip()
    return str(v.get("header_error", ""))


def _first_nonempty(*vals: Any) -> str:
    for v in vals:
        if v is None:
            continue
        s = str(v)
        if s and s.lower() not in {"none", "nan", "unknown"}:
            return s
    return ""


def _parse_cbar_source(flux_method: str, branch: str) -> str:
    s = flux_method.lower()
    if "source_cbar_field" in s:
        return "source_cbar_field"
    if "weighted_mean_from_source_cs" in s:
        return "weighted_mean_from_source_cs"
    if "current" in s or "integral" in s or "replay" in s:
        return "current_integral_or_replay_capacity_formula"
    if branch == "D15-RG_REPAIR_FROM_SOURCE_SOFTLABEL_BRANCH":
        return "source_cbar_field_or_weighted_mean_from_source_cs__not_explicit_in_summary"
    if branch == "D15-P4D_FULL_REPLAY_CURRENT_INTEGRAL_BRANCH":
        return "current_integral_from_replay_or_fixed_capacity_config"
    return "unknown"


def _parse_j_source(flux_method: str, branch: str) -> str:
    s = flux_method.lower()
    if "source_j_field" in s:
        return "source_j_field_from_npz"
    if "inferred_j_from_cbar_derivative" in s or "inferred" in s:
        return "inferred_from_cbar_derivative"
    if "current" in s or "replay" in s:
        return "current_boundary_formula"
    if branch == "D15-RG_REPAIR_FROM_SOURCE_SOFTLABEL_BRANCH":
        return "source_j_field_or_inferred_from_cbar_derivative__not_explicit_in_summary"
    if branch == "D15-P4D_FULL_REPLAY_CURRENT_INTEGRAL_BRANCH":
        return "current_boundary_formula"
    return "unknown"


def infer_semantic_branch(record: Mapping[str, Any], summary: Mapping[str, Any], key_info: Mapping[str, Any], scalars: Mapping[str, Any]) -> Dict[str, Any]:
    keys = set(key_info.get("keys", []))
    stage_text = _first_nonempty(summary.get("stage"), summary.get("source_stage"), record.get("source_stage"), scalars.get("radial_solver_version"))
    stage_low = stage_text.lower()
    source_npz_text = _first_nonempty(summary.get("source_npz"), summary.get("source_profile_npz"), scalars.get("source_profile_npz"))
    source_low = source_npz_text.lower()

    has_source_v1 = bool(key_info.get("has_source_v1_states")) or any(k.startswith("cs_a_source") or k.startswith("cs_c_source") for k in keys)
    has_rg = bool(key_info.get("has_rg_diagnostics")) or "P2Dlite-RG".lower() in stage_low
    if "p4d" in stage_low or "p4d" in source_low or "full_generate_one" in source_low:
        branch = "D15-P4D_FULL_REPLAY_CURRENT_INTEGRAL_BRANCH"
    elif has_source_v1 or "p0" in stage_low or "p3c" in stage_low or "p4b" in stage_low or "source" in stage_low:
        branch = "D15-RG_REPAIR_FROM_SOURCE_SOFTLABEL_BRANCH"
    elif has_rg:
        branch = "D15-RG_MIXED_OR_CONSOLIDATED_BRANCH"
    else:
        branch = "UNKNOWN_OR_MIXED_BRANCH"

    flux_a = _first_nonempty(summary.get("flux_method_a"), summary.get("source_flux_method_a"), scalars.get("source_flux_method_a"))
    flux_c = _first_nonempty(summary.get("flux_method_c"), summary.get("source_flux_method_c"), scalars.get("source_flux_method_c"))
    cbar_a = _parse_cbar_source(flux_a, branch)
    cbar_c = _parse_cbar_source(flux_c, branch)
    j_a = _parse_j_source(flux_a, branch)
    j_c = _parse_j_source(flux_c, branch)

    voltage_preserved = summary.get("voltage_labels_preserved")
    if voltage_preserved is None:
        voltage_preserved = summary.get("phis_c_voltage_preserved_from_source")
    if voltage_preserved is None and "phis_c_voltage_preserved_from_source" in scalars:
        voltage_preserved = scalars.get("phis_c_voltage_preserved_from_source")
    if isinstance(voltage_preserved, str):
        voltage_preserved_bool = voltage_preserved.strip().lower() in {"true", "1", "yes"}
    else:
        voltage_preserved_bool = bool(voltage_preserved) if voltage_preserved is not None else False

    phi_preserved = summary.get("phi_labels_preserved")
    if isinstance(phi_preserved, str):
        phi_preserved_bool = phi_preserved.strip().lower() in {"true", "1", "yes"}
    else:
        phi_preserved_bool = bool(phi_preserved) if phi_preserved is not None else False

    if branch == "D15-P4D_FULL_REPLAY_CURRENT_INTEGRAL_BRANCH":
        phis_src = "voltage_exp_direct_passthrough_or_p4d_voltage_formula"
        phie_src = "ohmic_current_lumped_formula_or_p4d_transport_proxy"
        theta0_source = "fixed_theta_initial_and_current_integral_in_p4d_config"
    elif branch == "D15-RG_REPAIR_FROM_SOURCE_SOFTLABEL_BRANCH":
        phis_src = "preserved_from_source_p2dlite_v1" if voltage_preserved_bool else "preserved_from_source_or_source_voltage_wrapper__not_explicit"
        phie_src = "preserved_from_source_p2dlite_v1" if phi_preserved_bool else "preserved_from_source_or_source_lumped_phi__not_explicit"
        theta0_source = "inherited_from_source_softlabel_state"
    elif branch == "D15-RG_MIXED_OR_CONSOLIDATED_BRANCH":
        phis_src = "consolidated_rg_softlabel_semantics__requires_branch_map"
        phie_src = "consolidated_rg_softlabel_semantics__requires_branch_map"
        theta0_source = "consolidated_or_inherited_state"
    else:
        phis_src = "unknown"
        phie_src = "unknown"
        theta0_source = "unknown"

    semantic_fields = [branch, cbar_a, cbar_c, j_a, j_c, phis_src, phie_src, theta0_source]
    unknowns = [x for x in semantic_fields if "unknown" in str(x).lower()]
    known_enough = branch != "UNKNOWN_OR_MIXED_BRANCH" and len(unknowns) == 0
    return {
        "stage": stage_text,
        "semantic_branch": branch,
        "cbar_source_a": cbar_a,
        "cbar_source_c": cbar_c,
        "J_source_a": j_a,
        "J_source_c": j_c,
        "phis_c_source_semantics": phis_src,
        "phie_source_semantics": phie_src,
        "theta0_inventory_source_semantics": theta0_source,
        "voltage_labels_preserved": voltage_preserved_bool,
        "phi_labels_preserved": phi_preserved_bool,
        "source_npz_text": source_npz_text,
        "source_flux_method_a_raw": flux_a,
        "source_flux_method_c_raw": flux_c,
        "semantic_known_enough": known_enough,
        "unknown_semantic_count": len(unknowns),
    }


def load_split_records(split_manifest: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    data = json_load(split_manifest, default={})
    if not isinstance(data, Mapping):
        raise ValueError(f"split_manifest is not a JSON object: {split_manifest}")
    records = data.get("records", [])
    if not isinstance(records, list):
        records = []
    return [dict(r) for r in records if isinstance(r, Mapping)], dict(data)


def select_records(records: Sequence[Mapping[str, Any]], profile_limit: int = 0, include_flagged_probe: bool = True) -> List[Mapping[str, Any]]:
    selected: List[Mapping[str, Any]] = []
    for r in records:
        if not include_flagged_probe and str(r.get("split")) == "flagged_probe":
            continue
        selected.append(r)
    if profile_limit and profile_limit > 0:
        return selected[:profile_limit]
    return selected


def audit_profiles(records: Sequence[Mapping[str, Any]], softlabel_root: Optional[Path], out_dir: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    split_counts = Counter()
    branch_counts = Counter()
    cbar_counts = Counter()
    j_counts = Counter()
    phie_counts = Counter()
    phis_counts = Counter()
    missing_npz: List[Dict[str, Any]] = []
    missing_keys_counter = Counter()
    known_semantics = 0
    total_with_npz = 0
    header_samples: Dict[str, Any] = {}

    for idx, r in enumerate(records):
        split = str(r.get("split", "UNKNOWN"))
        split_counts[split] += 1
        canonical = canonical_id_for_record(r)
        npz_path = resolve_npz_path(r, softlabel_root)
        base: Dict[str, Any] = {
            "index": idx,
            "split": split,
            "cell_uid": str(r.get("cell_uid", "")),
            "canonical_cell_uid": canonical,
            "batch": str(r.get("batch", "")),
            "protocol": str(r.get("protocol", "")),
            "battery": str(r.get("battery", "")),
            "softlabel_npz": str(npz_path) if npz_path else "",
            "replay_npz": str(r.get("replay_npz", "")),
            "manifest_source_stage": str(r.get("source_stage", "")),
        }
        if npz_path is None or not npz_path.exists():
            miss = {**base, "npz_exists": False, "status": "MISSING_SOFTLABEL_NPZ"}
            missing_npz.append(miss)
            rows.append(miss)
            continue
        total_with_npz += 1
        headers, header_error = read_npz_header_shapes(npz_path)
        keys = sorted(headers.keys())
        key_info = infer_from_keys(keys)
        key_info["keys"] = keys
        scalars = read_npz_safe_scalars(npz_path, keys)
        summary_path = find_sidecar_summary(r, npz_path)
        summary = json_load(summary_path, default={}) if summary_path else {}
        if not isinstance(summary, Mapping):
            summary = {}
        semantics = infer_semantic_branch(r, summary, key_info, scalars)
        branch_counts[semantics["semantic_branch"]] += 1
        cbar_counts[(semantics["cbar_source_a"], semantics["cbar_source_c"])] += 1
        j_counts[(semantics["J_source_a"], semantics["J_source_c"])] += 1
        phie_counts[semantics["phie_source_semantics"]] += 1
        phis_counts[semantics["phis_c_source_semantics"]] += 1
        if semantics.get("semantic_known_enough"):
            known_semantics += 1
        for mk in key_info.get("missing_required_state_keys", []):
            missing_keys_counter[mk] += 1
        row = {
            **base,
            "npz_exists": True,
            "npz_header_error": header_error or "",
            "summary_json": str(summary_path) if summary_path else "",
            "summary_stage": str(summary.get("stage", "")),
            "npz_key_count": len(keys),
            "required_state_keys_ok": bool(key_info["has_required_state_keys"]),
            "missing_required_state_keys": ";".join(key_info["missing_required_state_keys"]),
            "has_rg_diagnostics": bool(key_info["has_rg_diagnostics"]),
            "has_source_v1_states": bool(key_info["has_source_v1_states"]),
            "has_source_flux_method_keys": bool(key_info["has_source_flux_method_keys"]),
            "has_radial_solver_version": bool(key_info["has_radial_solver_version"]),
            "cs_a_shape_dtype": shape_str(headers, "cs_a"),
            "cs_c_shape_dtype": shape_str(headers, "cs_c"),
            "theta_a_shape_dtype": shape_str(headers, "theta_a"),
            "theta_c_shape_dtype": shape_str(headers, "theta_c"),
            "phie_shape_dtype": shape_str(headers, "phie"),
            "phis_c_shape_dtype": shape_str(headers, "phis_c"),
            "cbar_a_shape_dtype": shape_str(headers, "cbar_a"),
            "cbar_c_shape_dtype": shape_str(headers, "cbar_c"),
            "scalar_metadata_json": json.dumps(scalars, ensure_ascii=False, sort_keys=True),
            **semantics,
            "status": "PASS" if key_info["has_required_state_keys"] and semantics.get("semantic_known_enough") else "REVIEW",
        }
        rows.append(row)
        if len(header_samples) < 8:
            header_samples[canonical] = {
                "softlabel_npz": str(npz_path),
                "summary_json": str(summary_path) if summary_path else "",
                "keys_sample": keys[:80],
                "scalars": scalars,
                "headers_subset": {k: headers[k] for k in keys[:60]},
            }

    csv_path = out_dir / "D17_G0_PROFILE_SEMANTICS.csv"
    write_csv(rows, csv_path)
    json_dump(header_samples, out_dir / "D17_G0_NPZ_HEADER_SAMPLES.json")
    semantics_fraction = known_semantics / max(total_with_npz, 1)
    required_key_missing_profiles = sum(1 for row in rows if row.get("npz_exists") and row.get("required_state_keys_ok") is False)
    return {
        "profile_count_requested": len(records),
        "profile_count_audited": len(rows),
        "profile_count_with_npz": total_with_npz,
        "split_counts": dict(split_counts),
        "semantic_branch_counts": dict(branch_counts),
        "cbar_semantics_counts": {str(k): v for k, v in cbar_counts.items()},
        "J_semantics_counts": {str(k): v for k, v in j_counts.items()},
        "phie_semantics_counts": dict(phie_counts),
        "phis_c_semantics_counts": dict(phis_counts),
        "missing_npz_count": len(missing_npz),
        "missing_npz_examples": missing_npz[:20],
        "missing_required_key_profile_count": required_key_missing_profiles,
        "missing_required_key_counts": dict(missing_keys_counter),
        "semantics_known_fraction": semantics_fraction,
        "semantic_known_profile_count": known_semantics,
        "profile_semantics_csv": str(csv_path),
        "status": "PASS" if len(missing_npz) == 0 and required_key_missing_profiles == 0 else "REVIEW",
    }


def make_recommendations(code_scan: Mapping[str, Any], profile_audit: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    recs: List[str] = [
        "D17-G1 should implement a generator-code-equivalent supervised surrogate, not a voltage-only inverse PINN.",
        "Wrap or import gv1.p2dlite_rg.radial_solver.generate_rg_profile / infer_surface_flux_from_cbar semantics instead of rewriting radial diffusion from memory.",
        "For D15-RG repair-from-source branches, treat cs/theta as RG-repaired states and treat phie/phis_c as source-preserved or source-wrapper labels according to branch semantics.",
        "For D15-P4D full replay branches, treat phis_c as voltage-derived and phie as an ohmic/lumped-current transport proxy unless the local generator code proves otherwise.",
        "D17-G1 may use train-cell soft labels as supervised generator targets, but validation/frozen-test soft labels must remain withheld from training and structural tuning.",
        "G1 should start with 12 train / 3 validation / 3 frozen smoke before ALL55; promotion must be held-out-cell state metrics, not closed-set precision.",
    ]
    if code_scan.get("missing_files"):
        recs.append(f"Local repo is missing expected generator-related files: {code_scan.get('missing_files')}. Confirm local branch before G1.")
    if profile_audit.get("missing_npz_count", 0):
        recs.append("Some manifest records could not resolve solution_softlabels.npz; fix manifest/softlabel_root before G1.")
    md = [
        "# D17-G0 Generator Equivalence Audit Recommendations",
        "",
        "D17-G0 is audit-only. It does not train, modify checkpoints, or select models.",
        "",
        "## Recommendations",
        "",
    ]
    md.extend([f"{i}. {r}" for i, r in enumerate(recs, 1)])
    md.extend([
        "",
        "## Expected next package",
        "",
        "D17-G1: generator-code-equivalent supervised surrogate smoke. Train-cell soft labels are allowed; validation/frozen-test labels are withheld except for evaluation.",
    ])
    p = out_dir / "D17_G0_RECOMMENDATIONS.md"
    p.write_text("\n".join(md), encoding="utf-8")
    return {"recommendations": recs, "recommendations_md": str(p)}


def run_g0_audit(
    project_root: Path,
    split_manifest: Path,
    softlabel_root: Optional[Path],
    out_dir: Path,
    config: Optional[Mapping[str, Any]] = None,
    profile_limit: int = 0,
    include_flagged_probe: bool = True,
) -> Dict[str, Any]:
    config = dict(config or {})
    out_dir.mkdir(parents=True, exist_ok=True)
    generator_files = config.get("generator_code_files") or GENERATOR_FILES_DEFAULT
    min_known = float(config.get("min_semantics_known_fraction", 0.75))
    code_scan = scan_generator_code(project_root, generator_files)
    json_dump(code_scan, out_dir / "D17_G0_GENERATOR_CODE_SCAN.json")

    records, manifest = load_split_records(split_manifest)
    selected = select_records(records, profile_limit=profile_limit, include_flagged_probe=include_flagged_probe)
    profile_audit = audit_profiles(selected, softlabel_root=softlabel_root, out_dir=out_dir)
    manifest_summary = {
        "split_manifest": str(split_manifest),
        "manifest_hash_sha256": manifest.get("manifest_hash_sha256"),
        "counts": manifest.get("counts"),
        "seed": manifest.get("seed"),
        "flag_cell": manifest.get("flag_cell"),
        "record_count": len(records),
        "selected_record_count": len(selected),
    }
    json_dump(manifest_summary, out_dir / "D17_G0_SPLIT_MANIFEST_SUMMARY.json")
    recs = make_recommendations(code_scan, profile_audit, out_dir)

    reasons: List[str] = []
    blocker_reasons: List[str] = []
    if profile_audit.get("missing_npz_count", 0) > 0:
        blocker_reasons.append("some softlabel NPZ files cannot be resolved")
    if profile_audit.get("missing_required_key_profile_count", 0) > 0:
        blocker_reasons.append("some profiles miss required state keys")
    if not records:
        blocker_reasons.append("split manifest contains no records")
    if code_scan.get("status") not in {"PASS", "REVIEW"}:
        blocker_reasons.append("generator code scan failed")
    if profile_audit.get("semantics_known_fraction", 0.0) < min_known:
        reasons.append("known semantics fraction below threshold")
    if code_scan.get("status") == "REVIEW":
        reasons.append("generator code scan has review items")
    if profile_audit.get("status") == "REVIEW":
        reasons.append("profile audit has review items")

    status = "FAIL" if blocker_reasons else ("PASS" if not reasons else "REVIEW")
    g1_ready = status == "PASS"
    report = {
        "protocol": "D17-G0_GENERATOR_EQUIVALENCE_AUDIT",
        "created_at_utc": utc_now(),
        "status": status,
        "g1_ready": g1_ready,
        "reasons": blocker_reasons + reasons,
        "purpose": "Audit local D15 P2Dlite-RG generator code and ALL55 soft-label output semantics before D17-G supervised generator-surrogate training.",
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "state_softlabels_used_for_training": False,
        "state_softlabels_read_for_header_and_semantics_only": True,
        "project_root": str(project_root),
        "softlabel_root": str(softlabel_root) if softlabel_root else "",
        "out_dir": str(out_dir),
        "code_scan": code_scan,
        "split_manifest_summary": manifest_summary,
        "profile_audit": profile_audit,
        "recommendations": recs,
        "next_step": "D17-G1 generator-code-equivalent supervised surrogate smoke" if g1_ready else "Fix D17-G0 REVIEW/FAIL items before D17-G1",
    }
    json_dump(report, out_dir / "D17_G0_GENERATOR_EQUIVALENCE_AUDIT.json")
    return report
