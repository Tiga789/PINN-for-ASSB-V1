#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime helpers for D18 FORMAL55-DEPLOY 55-cell bounded operational audit."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

EXPECTED_STEP2_HASH = "3ff5b3c5afd53b931c772991a65512fa18d3db60766f9d11b458774108bde261"
DEFAULT_CELLS = [
    "Batch-5_random_walk_battery-7",
    "Batch-3_R2.5_battery-6",
    "Batch-6_GEO_battery-3",
]
TARGET_STATE_KEYS = {"cs_a", "cs_c", "theta_a", "theta_c"}
INITIAL_REFERENCE_PROTOCOLS = {"2C", "3C", "R2.5", "R3", "GEO"}
PROTOCOL_TO_BATCH = {"2C": 1, "3C": 2, "R2.5": 3, "R3": 4, "random_walk": 5, "GEO": 6}


_CANONICAL_UID_RE = re.compile(
    r"^Batch-(?P<batch>\d+)_(?P<protocol>2C|3C|R2\.5|R3|random_walk|GEO)_battery-(?P<battery>\d+)$",
    flags=re.IGNORECASE,
)
_BATTERY_TOKEN_RE = re.compile(r"(?i)(?:^|[^a-z0-9])battery[-_ ]?(\d+)(?=$|[^0-9])")
_BATCH_TOKEN_RE = re.compile(r"(?i)(?:^|[^a-z0-9])batch[-_ ]?(\d+)(?=$|[^0-9])")
_PROTOCOL_PATTERNS = (
    ("random_walk", re.compile(r"(?i)(?:^|[^a-z0-9])random[-_ ]?walk(?=$|[^a-z0-9])")),
    ("R2.5", re.compile(r"(?i)(?:^|[^a-z0-9])r2(?:\.|p|_)?5(?=$|[^a-z0-9])")),
    ("GEO", re.compile(r"(?i)(?:^|[^a-z0-9])geo(?=$|[^a-z0-9])")),
    ("R3", re.compile(r"(?i)(?:^|[^a-z0-9])r3(?=$|[^a-z0-9])")),
    ("3C", re.compile(r"(?i)(?:^|[^a-z0-9])3c(?=$|[^a-z0-9])")),
    ("2C", re.compile(r"(?i)(?:^|[^a-z0-9])2c(?=$|[^a-z0-9])")),
)


def _normalize_protocol(text: str) -> str:
    raw = str(text).strip()
    for name, pattern in _PROTOCOL_PATTERNS:
        if pattern.search(raw):
            return name
    return raw


def parse_canonical_uid(uid: str) -> Dict[str, Any]:
    """Parse the locked canonical UID grammar; reject fuzzy canonical IDs."""
    match = _CANONICAL_UID_RE.fullmatch(str(uid).strip())
    if not match:
        raise ValueError(f"Invalid canonical UID: {uid!r}")
    batch = int(match.group("batch"))
    protocol = _normalize_protocol(match.group("protocol"))
    battery = int(match.group("battery"))
    expected_batch = PROTOCOL_TO_BATCH[protocol]
    if batch != expected_batch:
        raise ValueError(
            f"Canonical UID batch/protocol mismatch: uid={uid!r}, "
            f"protocol={protocol}, batch={batch}, expected_batch={expected_batch}"
        )
    return {"batch": batch, "protocol": protocol, "battery": battery}


def _identity_tokens(texts: Sequence[str]) -> Dict[str, Any]:
    joined = " | ".join(str(x) for x in texts if str(x).strip())
    return {
        "joined": joined,
        "batteries": sorted({int(x) for x in _BATTERY_TOKEN_RE.findall(joined)}),
        "batches": sorted({int(x) for x in _BATCH_TOKEN_RE.findall(joined)}),
        "protocols": sorted({name for name, pattern in _PROTOCOL_PATTERNS if pattern.search(joined)}),
    }


def source_uid_compatibility(
    canonical_uid: str,
    source_uid: str,
    source_path: Path,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Reconcile legacy source names using explicit identity tokens only.

    The battery number must match, and at least one additional independent
    dimension (batch or protocol) must match. Any explicit conflict fails.
    No substring matching is used.
    """
    expected = parse_canonical_uid(canonical_uid)
    path = Path(source_path)
    observed = _identity_tokens([str(source_uid or ""), path.parent.name, path.stem])
    batteries = observed["batteries"]
    batches = observed["batches"]
    protocols = observed["protocols"]

    detail = {"expected": expected, "observed": observed}
    if not batteries:
        return False, "missing_battery_token", detail
    if batteries != [expected["battery"]]:
        return False, "battery_token_conflict", detail
    if batches and batches != [expected["batch"]]:
        return False, "batch_token_conflict", detail
    if protocols and protocols != [expected["protocol"]]:
        return False, "protocol_token_conflict", detail

    matched_dimensions = 1 + int(bool(batches)) + int(bool(protocols))
    if matched_dimensions < 2:
        return False, "insufficient_identity_dimensions", detail

    if batches and protocols:
        mode = "battery+batch+protocol"
    elif batches:
        mode = "battery+batch_legacy"
    else:
        mode = "battery+protocol_legacy"
    return True, mode, detail


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    def convert(x: Any) -> Any:
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [convert(v) for v in x]
        if isinstance(x, np.ndarray):
            return convert(x.tolist())
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, (np.floating, float)):
            v = float(x)
            return v if np.isfinite(v) else None
        if isinstance(x, Path):
            return str(x)
        return x
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(convert(obj), f, ensure_ascii=False, indent=2, allow_nan=False)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(str(key))
        fieldnames = keys
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def boolish(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "pass"}


def import_deploy_runtime(model_root: Path) -> Dict[str, Any]:
    runtime_dir = Path(model_root) / "runtime"
    if not runtime_dir.is_dir():
        raise FileNotFoundError(f"Runtime directory not found: {runtime_dir}")
    if str(runtime_dir) not in sys.path:
        sys.path.insert(0, str(runtime_dir))
    return {
        "adapter": importlib.import_module("formal55_adapter_runtime"),
        "core": importlib.import_module("_d18_step33_core"),
        "resume": importlib.import_module("_d18_step33_resume_core"),
        "step4": importlib.import_module("_d18_step4v2_core"),
        "step5": importlib.import_module("_d18_step5fix_core"),
    }


def discover_deploy_model(formal_root: Path, explicit: Optional[Path]) -> Tuple[Path, Path]:
    if explicit:
        root = Path(explicit)
        if root.is_file() and root.name == "D18_DEPLOY_READY_MANIFEST.json":
            manifest = root
            raw = Path(str(read_json(manifest).get("model_root", "")))
            root = raw if raw.is_dir() else manifest.parent / "MODELFIN_D18_FORMAL55_DEPLOY"
        else:
            manifest = Path("")
        if not root.is_dir():
            raise FileNotFoundError(f"Explicit deploy model root not found: {root}")
        return root, manifest

    candidates: List[Tuple[float, Path, Path]] = []
    seen: set[str] = set()
    for base in [formal_root / "Deploy_build", formal_root]:
        if not base.exists():
            continue
        for manifest in base.rglob("D18_DEPLOY_READY_MANIFEST.json"):
            key = str(manifest.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                data = read_json(manifest)
                if str(data.get("status", "")).upper() != "PASS" or not boolish(data.get("ready_for_operational_smoke")):
                    continue
                raw = Path(str(data.get("model_root", "")))
                root = raw if raw.is_dir() else manifest.parent / "MODELFIN_D18_FORMAL55_DEPLOY"
                if not root.is_dir():
                    matches = list(formal_root.rglob("MODELFIN_D18_FORMAL55_DEPLOY"))
                    if matches:
                        root = max(matches, key=lambda p: p.stat().st_mtime)
                if root.is_dir():
                    candidates.append((manifest.stat().st_mtime, root, manifest))
            except Exception:
                continue
    if not candidates:
        raise FileNotFoundError("No PASS D18 deploy-ready model found under FormalRoot")
    _, root, manifest = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
    return root, manifest


def registry_map(rows: Sequence[Mapping[str, str]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        uid = str(row.get("canonical_uid", "")).strip()
        if not uid:
            continue
        if uid in out:
            raise ValueError(f"Duplicate UID in registry: {uid}")
        out[uid] = dict(row)
    return out


def remap_source_path(raw: str, cache_root: Path) -> Path:
    p = Path(str(raw))
    if p.is_file():
        return p
    normalized = str(raw).replace("/", "\\")
    marker = "_gv1_cache\\"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        candidate = Path(cache_root) / Path(suffix.replace("\\", os.sep))
        if candidate.is_file():
            return candidate
    candidate = Path(cache_root) / "xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL" / "profiles" / p.parent.name / p.name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"External source not found: {raw}")

def _scalar_text(a: np.ndarray) -> str:
    return str(np.asarray(a).reshape(()).item())


def _normalize_1d(arr: np.ndarray, n: int, name: str) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim == 2 and 1 in x.shape:
        x = x.reshape(-1)
    if x.ndim != 1 or x.size != n:
        raise ValueError(f"{name} expected ({n},), observed={x.shape}")
    return x.astype(np.float32, copy=False)


def load_observable_profile(path: Path, core: Any) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        required = {"cbar_a", "cbar_c", "cycle_id", "t_global_s", "I_profile", "voltage_exp", "phie"}
        missing = sorted(required.difference(z.files))
        if "phis_c" not in z.files and "phis_c_soft" not in z.files:
            missing.append("phis_c|phis_c_soft")
        if missing:
            raise KeyError(f"Missing operational fields {missing}: {path}")
        n = int(np.asarray(z["cbar_a"]).reshape(-1).size)
        cycle_ids = core.normalize_cycle_ids(np.asarray(z["cycle_id"]), n)
        signals: Dict[str, np.ndarray] = {}
        for name in core.FEATURE_SIGNAL_VARS:
            if name in z.files:
                signals[name] = _normalize_1d(np.asarray(z[name]), n, name)
            elif name == "temperature_C":
                signals[name] = np.full(n, 25.0, dtype=np.float32)
            else:
                signals[name] = np.zeros(n, dtype=np.float32)
        cbar_a = _normalize_1d(np.asarray(z["cbar_a"]), n, "cbar_a")
        cbar_c = _normalize_1d(np.asarray(z["cbar_c"]), n, "cbar_c")
        phie = _normalize_1d(np.asarray(z["phie"]), n, "phie")
        phis_key = "phis_c" if "phis_c" in z.files else "phis_c_soft"
        phis_c = _normalize_1d(np.asarray(z[phis_key]), n, phis_key)
        r_a = np.asarray(z["r_a"], dtype=np.float64).reshape(-1) if "r_a" in z.files else np.linspace(0.0, 1.0, 17)
        r_c = np.asarray(z["r_c"], dtype=np.float64).reshape(-1) if "r_c" in z.files else np.linspace(0.0, 1.0, 17)
        source_uid = ""
        for key in ["cell_uid", "canonical_uid", "profile_uid", "uid"]:
            if key in z.files:
                try:
                    source_uid = _scalar_text(np.asarray(z[key]))
                    break
                except Exception:
                    pass
        present_targets = sorted(TARGET_STATE_KEYS.intersection(z.files))
    return {
        "n": n,
        "cycle_ids": cycle_ids,
        "signals": signals,
        "cbar_a": cbar_a,
        "cbar_c": cbar_c,
        "phie": phie,
        "phis_c": phis_c,
        "r_a": r_a,
        "r_c": r_c,
        "source_uid": source_uid,
        "target_state_arrays_present": present_targets,
        "target_state_arrays_loaded": False,
    }


def choose_indices(
    cycle_ids: np.ndarray,
    protocol: str,
    signals: Mapping[str, np.ndarray],
    step5: Any,
    step4: Any,
    cycles_per_cell: int,
    points_per_cycle: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[int]]:
    ranges = step5.contiguous_ranges(cycle_ids)
    eligible, excluded = step5.complete_noninitial_ranges(ranges, protocol)
    if not eligible:
        eligible = [r for r in ranges if not (protocol in INITIAL_REFERENCE_PROTOCOLS and r[0] == ranges[0][0])]
    if not eligible:
        raise RuntimeError(f"No eligible cycles for protocol={protocol}")
    count = min(max(1, int(cycles_per_cell)), len(eligible))
    positions = np.unique(np.round(np.linspace(0, len(eligible) - 1, count)).astype(int))
    chosen = [eligible[int(i)] for i in positions]
    parts: List[np.ndarray] = []
    ledger: List[Dict[str, Any]] = []
    for cid, start, stop in chosen:
        idx = step4.selected_indices(start, stop, signals, int(points_per_cycle))
        if idx.size:
            parts.append(idx)
            ledger.append({
                "cycle_id": int(cid),
                "start_idx": int(start),
                "stop_idx_exclusive": int(stop),
                "source_points": int(stop - start),
                "selected_points": int(idx.size),
            })
    if not parts:
        raise RuntimeError(f"No points selected for protocol={protocol}")
    return np.unique(np.concatenate(parts)), ledger, [int(x) for x in excluded]


def verify_bundle_manifest(model_root: Path) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    manifest = model_root / "manifests" / "artifact_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Artifact manifest not found: {manifest}")
    rows_out: List[Dict[str, Any]] = []
    failures: List[str] = []
    warnings: List[str] = []
    for row in read_csv(manifest):
        rel_text = str(row.get("relative_path", "")).replace("\\", "/")
        rel = Path(rel_text)
        p = model_root / rel
        expected_hash = str(row.get("sha256", "")).strip().lower()
        expected_size = int(float(row.get("size_bytes", -1) or -1))
        actual_hash = sha256_file(p) if p.is_file() else ""
        actual_size = p.stat().st_size if p.is_file() else -1
        ok = bool(p.is_file() and expected_hash and actual_hash.lower() == expected_hash and (expected_size < 0 or actual_size == expected_size))
        rows_out.append({
            "relative_path": rel_text,
            "expected_size_bytes": expected_size,
            "actual_size_bytes": actual_size,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "status": "PASS" if ok else "FAIL",
        })
        if not ok:
            failures.append(f"Bundle artifact mismatch: {p}")
        if "__pycache__" in rel_text or rel_text.endswith(".pyc"):
            warnings.append(f"Bundle contains cache artifact: {rel_text}")
    return rows_out, failures, warnings


def verify_parent_hashes(model_root: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    manifest = model_root / "manifests" / "parent_model_freeze_manifest.csv"
    rows_out: List[Dict[str, Any]] = []
    failures: List[str] = []
    for row in read_csv(manifest):
        rel = Path(str(row.get("bundle_relative_path", "")))
        p = model_root / rel
        expected = str(row.get("expected_sha256", "")).strip().lower()
        actual = sha256_file(p) if p.is_file() else ""
        ok = bool(p.is_file() and expected and actual.lower() == expected)
        rows_out.append({
            "artifact_kind": row.get("artifact_kind", ""),
            "bundle_relative_path": str(rel),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "status": "PASS" if ok else "FAIL",
        })
        if not ok:
            failures.append(f"Parent artifact hash mismatch: {p}")
    return rows_out, failures


def prepare_parent_runtime(model_root: Path, device: str, modules: Mapping[str, Any]) -> Dict[str, Any]:
    core, resume, step4 = modules["core"], modules["resume"], modules["step4"]
    cfg = model_root / "parent_models" / "config"
    calibrations = core.parse_calibration(cfg / "calibration_params.json")
    decisions = core.parse_step3fix_decisions(cfg / "step3fix_decision.csv")
    semantic = resume.load_semantic_models(cfg / "fit_only_semantic_radial_weights.json")
    dummy_provenance = model_root / "manifests" / "_not_present_f64_provenance.csv"
    f64_models, f64_audit, failures = core.load_f64_models(model_root / "parent_models" / "f64", dummy_provenance, device)
    if failures:
        raise RuntimeError("; ".join(failures))
    specialists, specialist_audit = resume.load_specialist_runtimes(model_root / "parent_models" / "step33", device)
    step34_runtimes = step4.load_step34_runtimes(model_root / "parent_models" / "step34", device)
    return {
        "calibrations": calibrations,
        "decisions": decisions,
        "semantic": semantic,
        "f64_models": f64_models,
        "f64_audit": f64_audit,
        "specialists": specialists,
        "specialist_audit": specialist_audit,
        "step34_runtimes": step34_runtimes,
        "lags": [1, 5, 30, 120, 600, 1800],
    }

def run_cell(
    *,
    uid: str,
    model_root: Path,
    source_path: Path,
    adapter_row: Mapping[str, str],
    confidence_row: Mapping[str, str],
    parent_runtime: Mapping[str, Any],
    modules: Mapping[str, Any],
    cycles_per_cell: int,
    points_per_cycle: int,
    inference_batch_size: int,
    radial_margin_fraction: float,
    out_dir: Path,
    save_sample_output: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    core = modules["core"]
    step4 = modules["step4"]
    step5 = modules["step5"]
    adapter_rt = modules["adapter"]

    protocol = str(adapter_row["protocol"])
    branch = str(adapter_row["branch"])
    batch = uid.split("_")[0]
    profile = load_observable_profile(source_path, core)
    idx, cycle_ledger, excluded_cycles = choose_indices(
        profile["cycle_ids"], protocol, profile["signals"], step5, step4,
        cycles_per_cell, points_per_cycle,
    )
    n = int(profile["n"])
    r_a, r_c = profile["r_a"], profile["r_c"]
    nr_a, nr_c = int(r_a.size), int(r_c.size)
    if nr_a != 17 or nr_c != 17:
        raise ValueError(f"Adapters require 17 radial points; observed a={nr_a}, c={nr_c}, uid={uid}")

    semantic = parent_runtime["semantic"]
    old_w_a = np.asarray(semantic[(branch, "anode")].weights, dtype=np.float64)
    old_w_c = np.asarray(semantic[(branch, "cathode")].weights, dtype=np.float64)
    can_w_a = step4.canonical_weights(nr_a, r_a)
    can_w_c = step4.canonical_weights(nr_c, r_c)
    calibration = parent_runtime["calibrations"][protocol]
    route_name, route_kind = step5.parent_route(protocol)
    candidates = step4.base_and_seed_candidates(
        idx=idx,
        n=n,
        signals=profile["signals"],
        cycle_ids=profile["cycle_ids"],
        cbar_a=profile["cbar_a"],
        cbar_c=profile["cbar_c"],
        protocol=protocol,
        branch=branch,
        batch=batch,
        old_w_a=old_w_a,
        old_w_c=old_w_c,
        can_w_a=can_w_a,
        can_w_c=can_w_c,
        calibration=calibration,
        step3fix_decisions=parent_runtime["decisions"],
        f64_models=parent_runtime["f64_models"],
        old_specialists=parent_runtime["specialists"],
        step34_runtimes=parent_runtime["step34_runtimes"],
        route_name=route_name,
        lags=parent_runtime["lags"],
        inference_batch_size=int(inference_batch_size),
        radial_margin_fraction=float(radial_margin_fraction),
        nr_a=nr_a,
        nr_c=nr_c,
    )
    use_learned = route_kind == "learned" and bool(candidates["learned_available"])
    parent_a = np.asarray(candidates["ensemble_a"] if use_learned else candidates["base_a"], dtype=np.float32)
    parent_c = np.asarray(candidates["ensemble_c"] if use_learned else candidates["base_c"], dtype=np.float32)
    parent_route_used = f"{route_name}:{'learned_ensemble' if use_learned else 'frozen_base'}"

    ranges = step5.contiguous_ranges(profile["cycle_ids"])
    phase, age, _ = step5.cycle_phase_and_age(profile["cycle_ids"], ranges)
    q_signed, q_abs, di_norm = step5.cumulative_features(profile["signals"])
    scalar = step5.build_scalar_features(idx, profile["signals"], phase, age, q_signed, q_abs, di_norm)

    adapter_path = model_root / Path(str(adapter_row["adapter_relative_path"]))
    expected_hash = str(adapter_row["adapter_sha256"]).strip().lower()
    actual_hash = sha256_file(adapter_path)
    if actual_hash.lower() != expected_hash:
        raise RuntimeError(f"Adapter hash mismatch for {uid}")
    cell_adapter = adapter_rt.load_cell_adapter(adapter_path)
    if cell_adapter.uid != uid:
        raise ValueError(f"Adapter UID mismatch: expected={uid}, observed={cell_adapter.uid}")

    dev_a = cell_adapter.anode.apply(parent_a, scalar, can_w_a)
    dev_c = cell_adapter.cathode.apply(parent_c, scalar, can_w_c)
    dev_a, scale_a = core.cap_radial_dev(
        dev_a, profile["cbar_a"][idx], float(calibration["anode"]["csmax"]),
        float(calibration["anode"]["radial_q995_theta"]), radial_margin_fraction, can_w_a,
    )
    dev_c, scale_c = core.cap_radial_dev(
        dev_c, profile["cbar_c"][idx], float(calibration["cathode"]["csmax"]),
        float(calibration["cathode"]["radial_q995_theta"]), radial_margin_fraction, can_w_c,
    )
    dev_a = np.asarray(dev_a, dtype=np.float32)
    dev_c = np.asarray(dev_c, dtype=np.float32)
    cbar_a = profile["cbar_a"][idx].astype(np.float32, copy=False)
    cbar_c = profile["cbar_c"][idx].astype(np.float32, copy=False)
    cs_a = cbar_a[:, None] + dev_a
    cs_c = cbar_c[:, None] + dev_c
    csmax_a = float(calibration["anode"]["csmax"])
    csmax_c = float(calibration["cathode"]["csmax"])
    theta_a = cs_a / np.float32(csmax_a)
    theta_c = cs_c / np.float32(csmax_c)
    phie = profile["phie"][idx].astype(np.float32, copy=False)
    phis_c = profile["phis_c"][idx].astype(np.float32, copy=False)

    finite_ok = all(np.all(np.isfinite(v)) for v in [cs_a, cs_c, theta_a, theta_c, phie, phis_c])
    zero_a = float(np.max(np.abs(np.sum(dev_a.astype(np.float64) * can_w_a[None, :], axis=1))) / max(csmax_a, 1e-12))
    zero_c = float(np.max(np.abs(np.sum(dev_c.astype(np.float64) * can_w_c[None, :], axis=1))) / max(csmax_c, 1e-12))
    cbar_err_a = float(np.max(np.abs(np.sum(cs_a.astype(np.float64) * can_w_a[None, :], axis=1) - cbar_a)) / max(csmax_a, 1e-12))
    cbar_err_c = float(np.max(np.abs(np.sum(cs_c.astype(np.float64) * can_w_c[None, :], axis=1) - cbar_c)) / max(csmax_c, 1e-12))
    theta_min = min(float(np.min(theta_a)), float(np.min(theta_c)))
    theta_max = max(float(np.max(theta_a)), float(np.max(theta_c)))
    theta_ok = theta_min >= -1e-5 and theta_max <= 1.0 + 1e-5
    source_uid = str(profile["source_uid"] or "")
    source_uid_ok, source_uid_mode, source_uid_detail = source_uid_compatibility(uid, source_uid, source_path)
    source_uid_observed = json.dumps(
        {
            "embedded_uid": source_uid or None,
            "source_parent": source_path.parent.name,
            "match_mode": source_uid_mode,
            "parsed": source_uid_detail,
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    checks = [
        {"canonical_uid": uid, "check": "source_uid_compatible", "observed": source_uid_observed, "threshold": "strict battery + (batch or protocol), no conflicts", "status": "PASS" if source_uid_ok else "FAIL"},
        {"canonical_uid": uid, "check": "target_state_arrays_loaded", "observed": False, "threshold": False, "status": "PASS"},
        {"canonical_uid": uid, "check": "six_outputs_finite", "observed": finite_ok, "threshold": True, "status": "PASS" if finite_ok else "FAIL"},
        {"canonical_uid": uid, "check": "zero_mean_anode", "observed": zero_a, "threshold": 1e-5, "status": "PASS" if zero_a <= 1e-5 else "FAIL"},
        {"canonical_uid": uid, "check": "zero_mean_cathode", "observed": zero_c, "threshold": 1e-5, "status": "PASS" if zero_c <= 1e-5 else "FAIL"},
        {"canonical_uid": uid, "check": "cbar_reconstruction_anode", "observed": cbar_err_a, "threshold": 1e-5, "status": "PASS" if cbar_err_a <= 1e-5 else "FAIL"},
        {"canonical_uid": uid, "check": "cbar_reconstruction_cathode", "observed": cbar_err_c, "threshold": 1e-5, "status": "PASS" if cbar_err_c <= 1e-5 else "FAIL"},
        {"canonical_uid": uid, "check": "theta_bounds", "observed": f"[{theta_min:.8g},{theta_max:.8g}]", "threshold": "[0,1] +/- 1e-5", "status": "PASS" if theta_ok else "FAIL"},
        {"canonical_uid": uid, "check": "adapter_sha256", "observed": actual_hash, "threshold": expected_hash, "status": "PASS" if actual_hash == expected_hash else "FAIL"},
    ]
    status = "PASS" if all(r["status"] == "PASS" for r in checks) else "FAIL"

    sample_path_text = ""
    sample_output_bytes = 0
    if save_sample_output:
        sample_dir = out_dir / "sample_predictions"
        ensure_dir(sample_dir)
        sample_path = sample_dir / f"{uid}_bounded_audit.npz"
        metadata = {
            "stage": "D18-ALL55-BOUNDED-OPERATIONAL-AUDIT",
            "canonical_uid": uid,
            "protocol": protocol,
            "branch": branch,
            "confidence": confidence_row.get("confidence", ""),
            "parent_route": parent_route_used,
            "anode_adapter_route": cell_adapter.anode.selected_route,
            "cathode_adapter_route": cell_adapter.cathode.selected_route,
            "source_path": str(source_path),
            "target_state_arrays_present_but_not_loaded": profile["target_state_arrays_present"],
            "potential_policy": "frozen source/Step2 passthrough",
        }
        np.savez_compressed(
            sample_path,
            source_index=idx.astype(np.int64),
            t_global_s=profile["signals"]["t_global_s"][idx],
            cycle_id=profile["cycle_ids"][idx],
            I_profile=profile["signals"]["I_profile"][idx],
            voltage_exp=profile["signals"]["voltage_exp"][idx],
            temperature_C=profile["signals"]["temperature_C"][idx],
            r_a=r_a.astype(np.float32), r_c=r_c.astype(np.float32),
            cbar_a=cbar_a, cbar_c=cbar_c,
            parent_dev_a=parent_a, parent_dev_c=parent_c,
            adapted_dev_a=dev_a, adapted_dev_c=dev_c,
            cs_a=cs_a, cs_c=cs_c, theta_a=theta_a, theta_c=theta_c,
            phie=phie, phis_c=phis_c,
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
        sample_path_text = str(sample_path)
        sample_output_bytes = int(sample_path.stat().st_size)

    row = {
        "canonical_uid": uid,
        "protocol": protocol,
        "branch": branch,
        "confidence": confidence_row.get("confidence", ""),
        "status": status,
        "source_path": str(source_path),
        "source_uid_embedded": source_uid,
        "source_uid_match_mode": source_uid_mode,
        "source_uid_compatible": source_uid_ok,
        "source_size_bytes": source_path.stat().st_size,
        "source_points": n,
        "source_cycle_count": int(np.unique(profile["cycle_ids"]).size),
        "selected_cycle_count": len(cycle_ledger),
        "selected_point_count": int(idx.size),
        "selected_cycle_ids": ",".join(str(r["cycle_id"]) for r in cycle_ledger),
        "excluded_cycle_count": len(excluded_cycles),
        "parent_route": parent_route_used,
        "anode_adapter_route": cell_adapter.anode.selected_route,
        "cathode_adapter_route": cell_adapter.cathode.selected_route,
        "target_state_arrays_present": ",".join(profile["target_state_arrays_present"]),
        "target_state_arrays_loaded": False,
        "theta_min": theta_min,
        "theta_max": theta_max,
        "zero_mean_anode": zero_a,
        "zero_mean_cathode": zero_c,
        "cbar_error_anode": cbar_err_a,
        "cbar_error_cathode": cbar_err_c,
        "cap_scale_anode_min": float(np.min(scale_a)),
        "cap_scale_cathode_min": float(np.min(scale_c)),
        "sample_output": sample_path_text,
        "sample_output_bytes": sample_output_bytes,
    }
    route_rows: List[Dict[str, Any]] = []
    for item in cycle_ledger:
        route_rows.append({
            "canonical_uid": uid,
            "protocol": protocol,
            "confidence": confidence_row.get("confidence", ""),
            "cycle_id": item["cycle_id"],
            "selected_points": item["selected_points"],
            "parent_route": parent_route_used,
            "anode_adapter_route": cell_adapter.anode.selected_route,
            "cathode_adapter_route": cell_adapter.cathode.selected_route,
        })
    return row, checks, route_rows


def directory_size(root: Path) -> Tuple[int, int, int]:
    total = largest = count = 0
    for p in root.rglob("*"):
        if p.is_file():
            size = p.stat().st_size
            total += size
            largest = max(largest, size)
            count += 1
    return total, largest, count


def self_test() -> Dict[str, Any]:
    n, nr = 32, 17
    r = np.linspace(0.0, 1.0, nr)
    edges = np.empty(nr + 1)
    edges[1:-1] = 0.5 * (r[:-1] + r[1:])
    edges[0], edges[-1] = 0.0, 1.0
    w = edges[1:] ** 3 - edges[:-1] ** 3
    w /= np.sum(w)
    x = np.random.default_rng(42).normal(size=(n, nr)).astype(np.float32)
    mean = np.sum(x.astype(np.float64) * w[None, :], axis=1, keepdims=True)
    dev = x - mean.astype(np.float32)
    err = float(np.max(np.abs(np.sum(dev.astype(np.float64) * w[None, :], axis=1))))

    uid_cases = [
        ("Batch-3_R2.5_battery-6", "0014_battery-6_R2.5_battery-6", Path("Batch-3_battery-6/solution_softlabels.npz"), True),
        ("Batch-5_random_walk_battery-7", "Batch-5_battery-7", Path("Batch-5_battery-7/solution_softlabels.npz"), True),
        ("Batch-6_GEO_battery-3", "Batch-6_battery-3", Path("Batch-6_battery-3/solution_softlabels.npz"), True),
        ("Batch-3_R2.5_battery-6", "Batch-3_battery-10", Path("Batch-3_battery-10/solution_softlabels.npz"), False),
        ("Batch-3_R2.5_battery-6", "Batch-4_battery-6", Path("Batch-4_battery-6/solution_softlabels.npz"), False),
        ("Batch-3_R2.5_battery-6", "battery-6", Path("unknown/solution_softlabels.npz"), False),
    ]
    uid_results: List[Dict[str, Any]] = []
    uid_ok = True
    for canonical, embedded, source_path, expected_ok in uid_cases:
        observed_ok, mode, detail = source_uid_compatibility(canonical, embedded, source_path)
        case_ok = observed_ok is expected_ok
        uid_ok = uid_ok and case_ok
        uid_results.append({
            "canonical_uid": canonical,
            "source_uid": embedded,
            "source_parent": source_path.parent.name,
            "expected_ok": expected_ok,
            "observed_ok": observed_ok,
            "mode": mode,
            "case_pass": case_ok,
        })

    status = "PASS" if err < 1e-6 and dev.shape == (n, nr) and uid_ok else "FAIL"
    return {
        "self_test": status,
        "max_zero_mean_error": err,
        "uid_resolver_test_pass": uid_ok,
        "uid_resolver_cases": uid_results,
    }
