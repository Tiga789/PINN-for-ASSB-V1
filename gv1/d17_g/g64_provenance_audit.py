from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.lib import format as npy_format


def read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: Sequence[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def sha256_file(path: str | Path, max_bytes: int = 0) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        remaining = int(max_bytes or 0)
        while True:
            if remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                remaining -= len(chunk)
            else:
                chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            if max_bytes and remaining <= 0:
                break
    return h.hexdigest()


def file_info(path: str | Path, hash_mode: str = "small") -> Dict[str, Any]:
    p = Path(path)
    out: Dict[str, Any] = {"path": str(p), "exists": p.exists()}
    if not p.exists():
        return out
    st = p.stat()
    out.update({"size_bytes": int(st.st_size), "mtime": float(st.st_mtime)})
    # Large softlabel npz can be tens/hundreds MB; default does not hash full content.
    if hash_mode == "full":
        out["sha256"] = sha256_file(p)
    elif hash_mode == "head1mb":
        out["sha256_head1mb"] = sha256_file(p, max_bytes=1024 * 1024)
    elif hash_mode == "small" and st.st_size <= 32 * 1024 * 1024:
        out["sha256"] = sha256_file(p)
    elif hash_mode == "small":
        out["sha256_head1mb"] = sha256_file(p, max_bytes=1024 * 1024)
        out["sha256_note"] = "large file: only first 1MB was hashed by default"
    return out


def flatten_json(obj: Any, prefix: str = "", max_list: int = 12) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_json(v, key, max_list=max_list))
    elif isinstance(obj, list):
        out[prefix + ".__len__" if prefix else "__len__"] = len(obj)
        for i, v in enumerate(obj[:max_list]):
            key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            out.update(flatten_json(v, key, max_list=max_list))
    else:
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            out[prefix] = obj
        else:
            out[prefix] = str(obj)
    return out


def grep_file(path: str | Path, patterns: Sequence[str], context_chars: int = 120) -> Dict[str, Any]:
    p = Path(path)
    res: Dict[str, Any] = {"path": str(p), "exists": p.exists(), "hits": {}}
    if not p.exists():
        return res
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    for pat in patterns:
        hits = []
        for i, line in enumerate(lines, start=1):
            if pat.lower() in line.lower():
                hits.append({"line": i, "text": line.strip()[:context_chars]})
        res["hits"][pat] = {"count": len(hits), "examples": hits[:8]}
    return res


def norm_text(s: Any) -> str:
    return str(s or "").replace("\\", "/").lower()


def row_text(row: Dict[str, Any]) -> str:
    return " ".join(norm_text(v) for v in row.values())


def split_needles(s: str) -> List[str]:
    toks = [t for t in re.split(r"[^A-Za-z0-9]+", str(s)) if t]
    return toks


def match_profile_rows(g0_csv: str | Path, split_manifest: str | Path, profile_contains: Sequence[str], branch_hint: str = "P4D") -> List[Dict[str, Any]]:
    rows = read_csv(g0_csv)
    if not rows:
        raise RuntimeError(f"No rows in G0 profile semantics CSV: {g0_csv}")

    chosen: List[Dict[str, str]] = []
    if profile_contains:
        for needle in profile_contains:
            matches = [r for r in rows if norm_text(needle) in row_text(r)]
            if not matches:
                toks = split_needles(needle)
                # Require most meaningful tokens, but ignore generic words.
                toks = [t for t in toks if t.lower() not in {"batch", "battery"}]
                matches = [r for r in rows if all(norm_text(t) in row_text(r) for t in toks)]
            if not matches:
                raise KeyError(f"No G0 profile row matches --profile_contains={needle!r}")
            if branch_hint:
                bmatches = [r for r in matches if norm_text(branch_hint) in norm_text(r.get("semantic_branch") or r.get("branch") or r.get("source_branch"))]
                if bmatches:
                    matches = bmatches
            chosen.append(matches[0])
    else:
        chosen = [r for r in rows if norm_text(branch_hint) in norm_text(r.get("semantic_branch") or r.get("branch") or r.get("source_branch"))]

    manifest = read_json(split_manifest)
    recs = manifest.get("records", []) if isinstance(manifest, dict) else []
    out: List[Dict[str, Any]] = []
    for r in chosen:
        d: Dict[str, Any] = dict(r)
        can = str(d.get("canonical_cell_uid") or d.get("canonical_cell_id") or d.get("cell_uid") or d.get("cell_id") or "")
        cell = str(d.get("cell_uid") or d.get("cell_id") or can)
        best = None
        for rr in recs:
            rt = row_text(rr)
            if (can and norm_text(can) in rt) or (cell and norm_text(cell) in rt):
                best = rr
                break
        if best is None:
            # Try token matching on canonical id if G0 canonical includes protocol and manifest has source id only.
            toks = split_needles(can or cell)
            toks = [t for t in toks if t.lower() not in {"batch", "battery"}]
            for rr in recs:
                rt = row_text(rr)
                if toks and all(norm_text(t) in rt for t in toks):
                    best = rr
                    break
        if best:
            for k, v in best.items():
                d.setdefault(k, v)
        d["canonical_cell_uid"] = d.get("canonical_cell_uid") or d.get("canonical_cell_id") or d.get("cell_uid") or d.get("cell_id") or ""
        d["cell_uid"] = d.get("cell_uid") or d.get("cell_id") or d["canonical_cell_uid"]
        d["semantic_branch"] = d.get("semantic_branch") or d.get("branch") or d.get("source_branch") or "UNKNOWN"
        out.append(d)
    return out


def first_existing_path(row: Dict[str, Any], names: Sequence[str], suffix: Optional[str] = None) -> Optional[Path]:
    for n in names:
        v = row.get(n)
        if v:
            p = Path(str(v))
            if p.exists():
                return p
            if suffix:
                q = p / suffix
                if q.exists():
                    return q
    return None


def path_candidate(row: Dict[str, Any], names: Sequence[str], suffix: Optional[str] = None) -> Optional[Path]:
    for n in names:
        v = row.get(n)
        if v:
            p = Path(str(v))
            if suffix and p.suffix.lower() != ".npz" and not p.name.endswith(".json"):
                # Treat as directory only if suffix is supplied and p does not already look like a file.
                q = p / suffix
                if q.exists():
                    return q
            if p.exists():
                return p
    return None


def npz_header_fast(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    out: Dict[str, Any] = {}
    with zipfile.ZipFile(p, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".npy"):
                continue
            key = name[:-4]
            try:
                with zf.open(name, "r") as fh:
                    version = npy_format.read_magic(fh)
                    if version == (1, 0):
                        shape, fortran, dtype = npy_format.read_array_header_1_0(fh)
                    elif version == (2, 0):
                        shape, fortran, dtype = npy_format.read_array_header_2_0(fh)
                    else:
                        shape, fortran, dtype = npy_format._read_array_header(fh, version)
                out[key] = {
                    "shape": list(shape),
                    "dtype": str(dtype),
                    "fortran_order": bool(fortran),
                    "size": int(np.prod(shape, dtype=np.int64)) if shape else 1,
                }
            except Exception as e:
                out[key] = {"error": repr(e)}
    return out


def jsonable_scalar(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return jsonable_scalar(x.item())
        if x.size <= 16:
            return [jsonable_scalar(v) for v in x.reshape(-1).tolist()]
        return {"array_shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", errors="replace")
        except Exception:
            return repr(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def npz_small_metadata(path: str | Path, header: Dict[str, Any], max_size: int = 64) -> Dict[str, Any]:
    small_keys = [k for k, v in header.items() if int(v.get("size", 10**18)) <= max_size]
    out: Dict[str, Any] = {}
    if not small_keys:
        return out
    with np.load(path, allow_pickle=True) as z:
        for k in small_keys:
            try:
                out[k] = jsonable_scalar(z[k])
            except Exception as e:
                out[k] = {"error": repr(e)}
    return out


def find_by_substrings(flat: Dict[str, Any], substrings: Sequence[str]) -> Dict[str, Any]:
    out = {}
    for k, v in flat.items():
        kt = norm_text(k)
        vt = norm_text(v)
        if any(norm_text(s) in kt or norm_text(s) in vt for s in substrings):
            out[k] = v
    return out


def extract_relevant_metadata(flat: Dict[str, Any]) -> Dict[str, Any]:
    groups = {
        "script_config_hash_path": ["script", "config", "sha", "hash", "source", "stage", "version", "provenance", "generator", "replay", "profile"],
        "theta_capacity_inventory": ["theta", "capacity", "qeff", "initial", "positive", "negative", "soc", "cbar", "csmax", "window"],
        "phie_phis_voltage": ["phie", "phis", "voltage", "ohmic", "current", "i_profile", "scale"],
        "radial_rg": ["radial", "rg", "fvm", "solver", "alpha", "gradient", "clip", "flux", "j_eff", "j_"],
        "time_cycle": ["time", "cycle", "step", "global", "grid", "sample"],
    }
    return {g: find_by_substrings(flat, subs) for g, subs in groups.items()}


def flatten_small_npz_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    return flatten_json(meta)


def scan_local_generator_code(project_root: str | Path, config: Dict[str, Any]) -> Dict[str, Any]:
    root = Path(project_root)
    files = config.get("generator_files", []) or [
        "scripts/d15_p4d_full_generate_one_rg_softlabel.py",
        "scripts/d15_p4d_generate_one_smoke_profile.py",
        "configs/d15_p4d_full_remaining14_config.json",
        "gv1/p2dlite_rg/radial_solver.py",
        "gv1/p2dlite_rg/io_utils.py",
    ]
    patterns = config.get("grep_patterns", []) or [
        "_cum_theta_from_current",
        "theta_positive_initial",
        "theta_negative_initial",
        "capacity_scale_Ah",
        "phie_ohmic_scale_V_per_A",
        "phis_c_soft",
        "voltage_exp",
        "cycle_id",
        "generate_rg_profile",
        "theta_c",
        "theta_a",
    ]
    scans = []
    for rel in files:
        p = root / rel
        info = file_info(p, hash_mode="full" if p.exists() and p.stat().st_size <= 4 * 1024 * 1024 else "head1mb")
        info["grep"] = grep_file(p, patterns) if p.exists() and p.is_file() else {"exists": p.exists(), "hits": {}}
        scans.append(info)
    # Also find nearby p4d configs/scripts that may not be in the static list.
    globs = config.get("extra_globs", []) or ["scripts/*p4d*.py", "configs/*p4d*.json", "configs/*D15*P4D*.json"]
    extra = []
    seen = {str(root / rel) for rel in files}
    for g in globs:
        for p in sorted(root.glob(g)):
            if str(p) in seen:
                continue
            info = file_info(p, hash_mode="full" if p.stat().st_size <= 4 * 1024 * 1024 else "head1mb")
            info["grep"] = grep_file(p, patterns) if p.is_file() else {"exists": p.exists(), "hits": {}}
            extra.append(info)
    return {"requested_files": scans, "extra_matched_files": extra}


def flatten_local_configs(project_root: str | Path, config: Dict[str, Any]) -> Dict[str, Any]:
    root = Path(project_root)
    globs = config.get("config_globs", []) or ["configs/*p4d*.json", "configs/*D15*P4D*.json"]
    out = {}
    for g in globs:
        for p in sorted(root.glob(g)):
            try:
                obj = read_json(p)
                flat = flatten_json(obj)
                interesting = extract_relevant_metadata(flat)
                out[str(p)] = {"file_info": file_info(p, hash_mode="full"), "interesting": interesting, "flat_selected": {k: v for k, v in flat.items() if any(s in norm_text(k) for s in ["theta", "capacity", "phie", "ohmic", "current", "initial", "scale"])}}
            except Exception as e:
                out[str(p)] = {"error": repr(e)}
    return out


def compare_profile_to_local(profile_meta_flat: Dict[str, Any], local_config_flat: Dict[str, Any]) -> Dict[str, Any]:
    # This is intentionally conservative: it does not claim equivalence from values alone.
    keys_of_interest = [
        "theta_positive_initial", "theta_negative_initial", "capacity_scale_Ah", "capacity_scale_ah", "phie_ohmic_scale_V_per_A", "phie_ohmic_scale_v_per_a",
        "theta_c_initial", "theta_a_initial", "theta_min", "theta_max",
    ]
    prof_hits = {}
    local_hits = {}
    for k, v in profile_meta_flat.items():
        kl = norm_text(k)
        for q in keys_of_interest:
            if norm_text(q) in kl:
                prof_hits[k] = v
    for path, obj in local_config_flat.items():
        flat_selected = obj.get("flat_selected", {}) if isinstance(obj, dict) else {}
        for k, v in flat_selected.items():
            kl = norm_text(k)
            for q in keys_of_interest:
                if norm_text(q) in kl:
                    local_hits[f"{path}::{k}"] = v
    return {"profile_hits": prof_hits, "local_config_hits": local_hits}


def audit_one_profile(row: Dict[str, Any], project_root: str | Path, hash_large: bool = False) -> Dict[str, Any]:
    soft_npz = path_candidate(row, ["softlabel_npz", "soft_label_npz", "solution_softlabels_npz"], None)
    if soft_npz is None:
        soft_npz = path_candidate(row, ["softlabel_dir", "soft_label_dir"], "solution_softlabels.npz")
    soft_summary = path_candidate(row, ["softlabel_summary", "soft_label_summary"], None)
    if soft_summary is None:
        soft_summary = path_candidate(row, ["softlabel_dir", "soft_label_dir"], "soft_label_summary.json")
    replay_npz = path_candidate(row, ["replay_npz", "replay_profile_npz", "source_replay_npz"], None)

    prof: Dict[str, Any] = {
        "canonical_cell_uid": row.get("canonical_cell_uid"),
        "cell_uid": row.get("cell_uid"),
        "split": row.get("split"),
        "protocol": row.get("protocol"),
        "semantic_branch": row.get("semantic_branch"),
        "paths": {
            "softlabel_npz": str(soft_npz) if soft_npz else "",
            "softlabel_summary": str(soft_summary) if soft_summary else "",
            "replay_npz": str(replay_npz) if replay_npz else "",
        },
        "file_info": {},
        "softlabel_summary_flat_selected": {},
        "softlabel_npz_header_selected": {},
        "softlabel_npz_scalar_metadata_selected": {},
        "replay_header_selected": {},
        "provenance_evidence_counts": {},
        "provenance_relevant": {},
        "warnings": [],
    }
    hmode = "full" if hash_large else "small"
    for name, p in [("softlabel_npz", soft_npz), ("softlabel_summary", soft_summary), ("replay_npz", replay_npz)]:
        prof["file_info"][name] = file_info(p, hash_mode=hmode) if p else {"exists": False}

    summary_flat = {}
    if soft_summary and soft_summary.exists():
        try:
            summary_flat = flatten_json(read_json(soft_summary))
            prof["softlabel_summary_flat_selected"] = extract_relevant_metadata(summary_flat)
        except Exception as e:
            prof["warnings"].append(f"Failed reading soft_label_summary.json: {e!r}")
    else:
        prof["warnings"].append("soft_label_summary.json not found")

    npz_flat = {}
    if soft_npz and soft_npz.exists():
        try:
            header = npz_header_fast(soft_npz)
            selected_header = {k: v for k, v in header.items() if any(s in norm_text(k) for s in ["theta", "cbar", "cs", "phie", "phis", "source", "version", "capacity", "initial", "scale", "voltage", "j_", "flux", "cycle", "time", "radial", "solver"])}
            prof["softlabel_npz_header_selected"] = selected_header
            small_meta = npz_small_metadata(soft_npz, header, max_size=64)
            npz_flat = flatten_small_npz_metadata(small_meta)
            prof["softlabel_npz_scalar_metadata_selected"] = extract_relevant_metadata(npz_flat)
        except Exception as e:
            prof["warnings"].append(f"Failed reading softlabel npz header/scalar metadata: {e!r}")
    else:
        prof["warnings"].append("solution_softlabels.npz not found")

    if replay_npz and replay_npz.exists():
        try:
            rheader = npz_header_fast(replay_npz)
            prof["replay_header_selected"] = {k: v for k, v in rheader.items() if any(s in norm_text(k) for s in ["time", "t_", "cycle", "step", "current", "i_", "voltage", "v_", "temperature", "temp"])}
        except Exception as e:
            prof["warnings"].append(f"Failed reading replay npz header: {e!r}")
    else:
        prof["warnings"].append("replay npz not found")

    combined_flat = {}
    combined_flat.update({f"summary::{k}": v for k, v in summary_flat.items()})
    combined_flat.update({f"npz_meta::{k}": v for k, v in npz_flat.items()})
    rel = extract_relevant_metadata(combined_flat)
    prof["provenance_relevant"] = rel

    evidence_terms = {
        "script_or_generator": ["script", "generator"],
        "config": ["config"],
        "hash_or_sha": ["sha", "hash"],
        "source_replay_or_profile": ["source", "replay", "profile"],
        "theta_capacity": ["theta", "capacity", "initial"],
        "phie_phis": ["phie", "phis", "voltage", "ohmic"],
        "radial_solver": ["radial", "solver", "rg", "fvm"],
    }
    counts = {}
    text_all = json.dumps(rel, ensure_ascii=False).lower()
    for group, terms in evidence_terms.items():
        counts[group] = int(any(t in text_all for t in terms))
    prof["provenance_evidence_counts"] = counts
    prof["provenance_score"] = int(sum(counts.values()))
    return prof


def summarize_recommendation(profile_audits: Sequence[Dict[str, Any]], local_scan: Dict[str, Any]) -> Tuple[str, List[str], bool]:
    blockers: List[str] = []
    if not profile_audits:
        return "FAIL_NO_PROFILES_SELECTED", ["No P4D/GEO profiles selected"], False
    # Basic file presence.
    for p in profile_audits:
        label = p.get("canonical_cell_uid") or p.get("cell_uid")
        if not p["file_info"].get("softlabel_npz", {}).get("exists"):
            blockers.append(f"{label}: solution_softlabels.npz missing")
        if not p["file_info"].get("softlabel_summary", {}).get("exists"):
            blockers.append(f"{label}: soft_label_summary.json missing")
    # Provenance score.
    low = [p for p in profile_audits if int(p.get("provenance_score", 0)) < 5]
    if low:
        labels = ", ".join(str(p.get("canonical_cell_uid") or p.get("cell_uid")) for p in low)
        blockers.append(f"provenance metadata incomplete for: {labels}")
    # Look for script/config/hash evidence specifically.
    no_script_hash = []
    for p in profile_audits:
        counts = p.get("provenance_evidence_counts", {})
        if not (counts.get("script_or_generator") and counts.get("config") and counts.get("hash_or_sha")):
            no_script_hash.append(str(p.get("canonical_cell_uid") or p.get("cell_uid")))
    if no_script_hash:
        blockers.append("missing explicit script/config/hash provenance in selected soft labels: " + ", ".join(no_script_hash))
    # Local code exists?
    req = local_scan.get("requested_files", [])
    missing_local = [x.get("path") for x in req if not x.get("exists")]
    if missing_local:
        blockers.append("local generator files missing: " + "; ".join(map(str, missing_local)))

    if blockers:
        return "STOP_PROVENANCE_INCOMPLETE_DO_NOT_TRAIN_OR_PATCH", blockers, False
    return "PROVENANCE_PRESENT_RUN_VERSION_MATCH_OR_REGEN_EQUIVALENCE_TEST", [], True


def run_provenance_audit(args: Any) -> Dict[str, Any]:
    t0 = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = read_json(args.config) if getattr(args, "config", None) and Path(args.config).exists() else {}

    selected = match_profile_rows(args.g0_profile_semantics_csv, args.split_manifest, args.profile_contains or [], branch_hint=cfg.get("branch_hint", "P4D"))
    local_scan = scan_local_generator_code(args.project_root, cfg)
    local_configs = flatten_local_configs(args.project_root, cfg)

    audits = []
    compare_rows: List[Dict[str, Any]] = []
    for row in selected:
        pa = audit_one_profile(row, args.project_root, hash_large=bool(getattr(args, "hash_large", False)))
        # Compare interesting profile metadata to local configs.
        flat_profile = {}
        flat_profile.update(flatten_json(pa.get("softlabel_summary_flat_selected", {})))
        flat_profile.update(flatten_json(pa.get("softlabel_npz_scalar_metadata_selected", {})))
        cmp = compare_profile_to_local(flat_profile, local_configs)
        pa["profile_vs_local_config_selected"] = cmp
        audits.append(pa)
        for group, sub in pa.get("provenance_relevant", {}).items():
            compare_rows.append({
                "canonical_cell_uid": pa.get("canonical_cell_uid"),
                "cell_uid": pa.get("cell_uid"),
                "protocol": pa.get("protocol"),
                "semantic_branch": pa.get("semantic_branch"),
                "metadata_group": group,
                "n_items": len(sub) if isinstance(sub, dict) else 0,
                "keys_preview": "; ".join(list(sub.keys())[:20]) if isinstance(sub, dict) else "",
            })

    recommendation, blockers, ready = summarize_recommendation(audits, local_scan)

    # Write detailed files.
    write_json(local_scan, out_dir / "D17_G64_LOCAL_GENERATOR_CODE_SCAN.json")
    write_json(local_configs, out_dir / "D17_G64_LOCAL_P4D_CONFIG_SCAN.json")
    write_json({"profiles": audits}, out_dir / "D17_G64_PROFILE_PROVENANCE_DETAILS.json")
    write_csv(compare_rows, out_dir / "D17_G64_PROFILE_PROVENANCE_INDEX.csv")

    summary = {
        "protocol": "D17-G6.4_P4D_GEO_PROVENANCE_AUDIT",
        "status": "PASS" if not any("missing" in b.lower() and "solution_softlabels" in b.lower() for b in blockers) else "REVIEW",
        "provenance_ready": bool(ready),
        "recommendation": recommendation,
        "blockers": blockers,
        "selected_profile_count": len(selected),
        "evaluated_profile_count": len(audits),
        "elapsed_s": float(time.perf_counter() - t0),
        "policy": {
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "large_state_arrays_loaded": False,
            "only_headers_small_scalar_metadata_and_sidecar_json_read": True,
            "do_not_train_or_patch_until_provenance_ready": True,
        },
        "profile_provenance_scores": [
            {
                "canonical_cell_uid": p.get("canonical_cell_uid"),
                "cell_uid": p.get("cell_uid"),
                "protocol": p.get("protocol"),
                "semantic_branch": p.get("semantic_branch"),
                "provenance_score": p.get("provenance_score"),
                "evidence_counts": p.get("provenance_evidence_counts"),
                "warnings": p.get("warnings"),
            }
            for p in audits
        ],
        "outputs": {
            "summary_json": str(out_dir / "D17_G64_P4D_PROVENANCE_AUDIT_SUMMARY.json"),
            "profile_details_json": str(out_dir / "D17_G64_PROFILE_PROVENANCE_DETAILS.json"),
            "local_code_scan_json": str(out_dir / "D17_G64_LOCAL_GENERATOR_CODE_SCAN.json"),
            "local_config_scan_json": str(out_dir / "D17_G64_LOCAL_P4D_CONFIG_SCAN.json"),
            "profile_provenance_index_csv": str(out_dir / "D17_G64_PROFILE_PROVENANCE_INDEX.csv"),
        },
    }
    write_json(summary, out_dir / "D17_G64_P4D_PROVENANCE_AUDIT_SUMMARY.json")
    return summary
