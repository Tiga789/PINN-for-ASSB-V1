from __future__ import annotations

import csv
import importlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import compact_exception, dump_json, sha256_file, utc_now_iso, write_csv


STATE_KEYS = ("theta_a", "theta_c", "cs_a", "cs_c", "phie", "phis_c")
BLOCKED_SPLITS = {"frozen_test", "test", "flagged_probe", "flagged", "probe"}


@dataclass
class DenseCasepackResult:
    status: str
    output_dir: Path
    case_files: list[Path]
    manifest_rows: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    summary: dict[str, Any]


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _split_name(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "val": "validation",
        "valid": "validation",
        "g2_train_internal_heldout": "internal_heldout",
        "train_internal_heldout": "internal_heldout",
        "heldout": "internal_heldout",
        "frozentest": "frozen_test",
        "flagged": "flagged_probe",
        "probe": "flagged_probe",
    }
    return aliases.get(text, text)


def _branch_family(branch: str, protocol: str) -> str:
    text = f"{branch} {protocol}".upper()
    if any(token in text for token in ("P4D", "CURRENT_INTEGRAL", "GEO", "RANDOM_WALK")):
        return "P4D"
    return "RG"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _internal_heldout_uids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    out: set[str] = set()
    for row in _read_csv(path):
        if _split_name(row.get("split")) == "internal_heldout":
            uid = str(row.get("canonical_cell_uid", "")).strip()
            if uid:
                out.add(_norm(uid))
    return out


def _find_cycle_key(files: Sequence[str]) -> str | None:
    for key in ("cycle_id", "cycle", "cycle_index"):
        if key in files:
            return key
    return None


def _unique_cycles(softlabel_npz: Path) -> list[int]:
    with np.load(softlabel_npz, allow_pickle=True) as z:
        key = _find_cycle_key(z.files)
        if key is None:
            raise KeyError(f"No cycle_id key in {softlabel_npz}")
        raw = np.asarray(z[key]).reshape(-1)
    vals: list[int] = []
    for value in raw:
        try:
            iv = int(float(value))
        except Exception:
            continue
        vals.append(iv)
    if not vals:
        raise ValueError(f"No numeric cycles in {softlabel_npz}")
    return sorted(set(vals))


def _window(values: Sequence[int], center_index: int, width: int) -> list[int]:
    if not values:
        return []
    width = max(1, int(width))
    start = max(0, center_index - width // 2)
    end = min(len(values), start + width)
    start = max(0, end - width)
    return list(values[start:end])


def choose_cycles(softlabel_npz: Path, case_cfg: Mapping[str, Any], default_width: int = 3) -> tuple[list[int], dict[str, list[int]]]:
    explicit = case_cfg.get("cycles")
    available = _unique_cycles(softlabel_npz)
    if explicit:
        if isinstance(explicit, str):
            requested: list[int] = []
            for part in explicit.replace("，", ",").split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    a, b = part.split("-", 1)
                    ia, ib = int(a), int(b)
                    if ib < ia:
                        ia, ib = ib, ia
                    requested.extend(range(ia, ib + 1))
                else:
                    requested.append(int(part))
        else:
            requested = [int(x) for x in explicit]
        selected = [x for x in sorted(set(requested)) if x in set(available)]
        if len(selected) < int(case_cfg.get("min_cycles", 3)):
            raise ValueError(f"Explicit cycles {requested} yield only {selected} for {softlabel_npz}")
        thirds = np.array_split(np.asarray(selected, dtype=int), 3)
        groups = {
            "early": [int(x) for x in thirds[0].tolist()],
            "middle": [int(x) for x in thirds[1].tolist()],
            "late": [int(x) for x in thirds[2].tolist()],
        }
        return selected, groups

    width = int(case_cfg.get("cycles_per_position", default_width))
    early = list(available[:width])
    middle = _window(available, len(available) // 2, width)
    late = list(available[-width:])
    selected = sorted(set(early + middle + late))
    if len(selected) < int(case_cfg.get("min_cycles", 3)):
        raise ValueError(f"Not enough cycles in {softlabel_npz}: {selected}")
    return selected, {"early": early, "middle": middle, "late": late}


def _cycle_position_vector(cycle_id: np.ndarray, groups: Mapping[str, Sequence[int]]) -> np.ndarray:
    out = np.full(cycle_id.size, "unknown", dtype="U12")
    for label in ("early", "middle", "late"):
        vals = np.asarray(list(groups.get(label, [])), dtype=np.int64)
        if vals.size:
            out[np.isin(cycle_id.astype(np.int64), vals)] = label
    return out


def _contiguous_diagnostic_time(source_time: np.ndarray, cycle_id: np.ndarray) -> np.ndarray:
    """Remove large gaps between non-contiguous selected cycles without changing within-cycle dt."""
    t = np.asarray(source_time, dtype=np.float64).reshape(-1)
    cyc = np.asarray(cycle_id, dtype=np.int64).reshape(-1)
    if t.size != cyc.size or t.size == 0:
        return t.astype(np.float32)
    out = np.zeros_like(t, dtype=np.float64)
    change = np.flatnonzero(cyc[1:] != cyc[:-1]) + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [t.size]])
    cursor = 0.0
    finite_dt = np.diff(t)
    finite_dt = finite_dt[np.isfinite(finite_dt) & (finite_dt > 0)]
    default_dt = float(np.median(finite_dt)) if finite_dt.size else 1.0
    for start, stop in zip(starts, stops):
        seg = t[start:stop]
        rel = seg - seg[0]
        rel = np.where(np.isfinite(rel) & (rel >= 0), rel, 0.0)
        out[start:stop] = cursor + rel
        cursor = float(out[stop - 1] + default_dt)
    return out.astype(np.float32)


def _save_real_case(
    *,
    case_cfg: Mapping[str, Any],
    record: Mapping[str, Any],
    sem_row: Mapping[str, str],
    data: Mapping[str, Any],
    pred_by_target: Mapping[str, np.ndarray],
    split: str,
    cycle_groups: Mapping[str, Sequence[int]],
    output_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    uid = str(record.get("canonical_cell_uid") or record.get("cell_uid") or case_cfg.get("id") or output_path.stem)
    cell_uid = str(record.get("cell_uid") or uid)
    branch = str(data.get("branch") or sem_row.get("semantic_branch") or "UNKNOWN")
    protocol = str(data.get("protocol") or record.get("protocol") or case_cfg.get("protocol") or "UNKNOWN")
    branch_family = _branch_family(branch, protocol)
    cycle_id = np.asarray(data["cycle_id"], dtype=np.int64).reshape(-1)
    step_type = data.get("step_type")
    if step_type is None:
        current = np.asarray(data["I"], dtype=np.float32)
        eps = max(1e-8, float(np.nanpercentile(np.abs(current), 99)) * 1e-3)
        st = np.full(current.size, "rest", dtype="U16")
        st[current > eps] = "charge"
        st[current < -eps] = "discharge"
        step_type = st
    else:
        step_type = np.asarray(step_type).reshape(-1).astype("U64")
    source_time = np.asarray(data["t"], dtype=np.float32)
    arrays: dict[str, Any] = {
        "t_global_s": _contiguous_diagnostic_time(source_time, cycle_id),
        "t_source_global_s": source_time,
        "cycle_id": cycle_id,
        "cycle_position": _cycle_position_vector(cycle_id, cycle_groups),
        "I_profile": np.asarray(data["I"], dtype=np.float32),
        "voltage_exp": np.asarray(data["V"], dtype=np.float32),
        "temperature_C": np.asarray(data["T"], dtype=np.float32),
        "step_type": step_type,
        "r_a": np.asarray(data["radial"]["a"], dtype=np.float32),
        "r_c": np.asarray(data["radial"]["c"], dtype=np.float32),
        "canonical_cell_uid": np.array(uid),
        "cell_uid": np.array(cell_uid),
        "split": np.array(split),
        "protocol": np.array(protocol),
        "semantic_branch": np.array(branch),
        "branch_family": np.array(branch_family),
        "case_role": np.array(str(case_cfg.get("role", "explicit_dense_diagnostic"))),
        "case_id": np.array(str(case_cfg.get("id", output_path.stem))),
        "casepack_version": np.array("D18-S1-DENSE-CASEPACK-FIX-v2"),
        "source_softlabel_npz": np.array(str(data.get("softlabel_npz", ""))),
        "source_replay_npz": np.array(str(data.get("replay_npz", ""))),
        "source_checkpoint": np.array(str(checkpoint_path)),
        "selected_cycles": np.asarray(data.get("selected_cycles", []), dtype=np.int64),
        "selected_cycle_groups_json": np.array(json.dumps({k: [int(x) for x in v] for k, v in cycle_groups.items()})),
    }
    targets = data.get("targets", {})
    for key in STATE_KEYS:
        if key in pred_by_target and key in targets:
            arrays[f"{key}_pred"] = np.asarray(pred_by_target[key], dtype=np.float32)
            arrays[f"{key}_true_report_only"] = np.asarray(targets[key], dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return {
        "case_id": str(case_cfg.get("id", output_path.stem)),
        "canonical_cell_uid": uid,
        "cell_uid": cell_uid,
        "split": split,
        "role": str(case_cfg.get("role", "explicit_dense_diagnostic")),
        "protocol": protocol,
        "semantic_branch": branch,
        "branch_family": branch_family,
        "selected_cycles": ",".join(str(x) for x in data.get("selected_cycles", [])),
        "cycle_count": len(set(int(x) for x in cycle_id.tolist())),
        "n_time": int(cycle_id.size),
        "prediction_npz": str(output_path),
        "prediction_sha256": sha256_file(output_path),
        "source_softlabel_npz": str(data.get("softlabel_npz", "")),
        "source_checkpoint": str(checkpoint_path),
        "status": "PASS",
    }


def _make_synthetic_case(case_cfg: Mapping[str, Any], output_path: Path, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_r = 17
    selected_cycles = np.asarray([1, 2, 3, 49, 50, 51, 98, 99, 100], dtype=np.int64)
    points_per_cycle = int(case_cfg.get("synthetic_points_per_cycle", 72))
    cycle_id = np.repeat(selected_cycles, points_per_cycle)
    n = cycle_id.size
    local = np.tile(np.linspace(0.0, 1.0, points_per_cycle, endpoint=False), selected_cycles.size)
    t = np.arange(n, dtype=np.float64) * 5.0
    phase = np.tile(np.concatenate([
        np.full(points_per_cycle // 3, "charge", dtype="U16"),
        np.full(points_per_cycle // 3, "rest", dtype="U16"),
        np.full(points_per_cycle - 2 * (points_per_cycle // 3), "discharge", dtype="U16"),
    ]), selected_cycles.size)
    I = np.zeros(n, dtype=np.float64)
    I[phase == "charge"] = 2.0
    I[phase == "discharge"] = -2.0
    protocol = str(case_cfg.get("protocol", "")).strip()
    if not protocol:
        batch = str(case_cfg.get("batch", ""))
        protocol = {"1": "2C", "2": "3C", "3": "R2.5", "4": "R3", "5": "random_walk", "6": "GEO"}.get(batch, "2C")
    branch_family = str(case_cfg.get("branch_family", _branch_family("", protocol)))
    branch = "P4D_CURRENT_INTEGRAL" if branch_family == "P4D" else "RG_REPAIRED"
    rho = np.linspace(0.0, 1.0, n_r)
    shape = rho**2 - 3.0 / 5.0
    cyc_norm = (cycle_id - cycle_id.min()) / max(1.0, float(cycle_id.max() - cycle_id.min()))
    base_a = 0.64 + 0.08 * np.sin(2 * np.pi * local) - 0.05 * cyc_norm
    base_c = 0.42 - 0.07 * np.sin(2 * np.pi * local) + 0.045 * cyc_norm
    amp = 0.02 + 0.012 * np.abs(I) / 2.0
    cs_a = base_a[:, None] + amp[:, None] * shape[None, :]
    cs_c = base_c[:, None] - amp[:, None] * shape[None, :]
    theta_a = cs_a.copy()
    theta_c = cs_c.copy()
    phie = (0.08 + 0.025 * I + 0.015 * np.sin(4 * np.pi * local))[:, None]
    phis_c = (3.72 + 0.28 * np.sin(2 * np.pi * local) + 0.055 * I - 0.09 * cyc_norm)[:, None]
    truths = {"cs_a": cs_a, "cs_c": cs_c, "theta_a": theta_a, "theta_c": theta_c, "phie": phie, "phis_c": phis_c}
    # Deliberately reproduce a structural full-cycle failure: aging drift, phase lag,
    # gauge drift and radial attenuation. This is only used by package self-test.
    lag = 7 + seed % 5
    preds: dict[str, np.ndarray] = {}
    for key, truth in truths.items():
        shifted = np.roll(truth, lag, axis=0)
        if key in {"cs_a", "theta_a"}:
            pred = 0.82 * shifted + (0.09 + 0.08 * cyc_norm)[:, None]
        elif key in {"cs_c", "theta_c"}:
            pred = 0.78 * shifted + (0.10 - 0.06 * cyc_norm)[:, None]
        elif key == "phie":
            pred = shifted + (0.08 + 0.12 * cyc_norm)[:, None]
        else:
            pred = 0.90 * shifted + (0.18 * cyc_norm - 0.05)[:, None]
        pred = pred + rng.normal(0.0, 0.002, size=pred.shape)
        preds[key] = pred
    positions = np.full(n, "middle", dtype="U12")
    positions[np.isin(cycle_id, selected_cycles[:3])] = "early"
    positions[np.isin(cycle_id, selected_cycles[-3:])] = "late"
    uid = str(case_cfg.get("uid", case_cfg.get("id", output_path.stem)))
    split = _split_name(case_cfg.get("split") or case_cfg.get("expected_split") or "validation")
    arrays: dict[str, Any] = {
        "t_global_s": t.astype(np.float32),
        "cycle_id": cycle_id,
        "cycle_position": positions,
        "I_profile": I.astype(np.float32),
        "voltage_exp": phis_c[:, 0].astype(np.float32),
        "temperature_C": np.full(n, 25.0, dtype=np.float32),
        "step_type": phase,
        "r_a": rho.astype(np.float32),
        "r_c": rho.astype(np.float32),
        "canonical_cell_uid": np.array(uid),
        "cell_uid": np.array(uid),
        "split": np.array(split),
        "protocol": np.array(protocol),
        "semantic_branch": np.array(branch),
        "branch_family": np.array(branch_family),
        "case_role": np.array(str(case_cfg.get("role", "synthetic_fixture"))),
        "case_id": np.array(str(case_cfg.get("id", output_path.stem))),
        "casepack_version": np.array("D18-S1-DENSE-CASEPACK-FIX-v2-SYNTHETIC"),
        "selected_cycles": selected_cycles,
        "selected_cycle_groups_json": np.array(json.dumps({"early": selected_cycles[:3].tolist(), "middle": selected_cycles[3:6].tolist(), "late": selected_cycles[6:].tolist()})),
    }
    for key in STATE_KEYS:
        arrays[f"{key}_true_report_only"] = truths[key].astype(np.float32)
        arrays[f"{key}_pred"] = preds[key].astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return {
        "case_id": str(case_cfg.get("id", output_path.stem)),
        "canonical_cell_uid": uid,
        "cell_uid": uid,
        "split": split,
        "role": str(case_cfg.get("role", "synthetic_fixture")),
        "protocol": protocol,
        "semantic_branch": branch,
        "branch_family": branch_family,
        "selected_cycles": ",".join(str(x) for x in selected_cycles),
        "cycle_count": int(selected_cycles.size),
        "n_time": int(n),
        "prediction_npz": str(output_path),
        "prediction_sha256": sha256_file(output_path),
        "source_softlabel_npz": "SYNTHETIC_FIXTURE",
        "source_checkpoint": "SYNTHETIC_FIXTURE",
        "status": "PASS",
    }


def build_dense_casepack(
    *,
    project_root: str | Path,
    output_dir: str | Path,
    config: Mapping[str, Any],
) -> DenseCasepackResult:
    project_root = Path(project_root).resolve()
    out = Path(output_dir).resolve()
    cases_dir = out / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    cfg = config.get("dense_casepack", {}) if isinstance(config.get("dense_casepack"), Mapping) else {}
    case_cfgs = list(cfg.get("cases", []))
    synthetic = bool(cfg.get("synthetic_fixture_mode", False))
    manifest_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    case_files: list[Path] = []

    if synthetic:
        for i, case_cfg in enumerate(case_cfgs):
            try:
                case_id = str(case_cfg.get("id", f"synthetic_{i:02d}"))
                path = cases_dir / f"{i:02d}_{case_id}.npz"
                row = _make_synthetic_case(case_cfg, path, seed=int(cfg.get("synthetic_seed", 18018)) + i)
                manifest_rows.append(row)
                case_files.append(path)
            except Exception as exc:
                failures.append({"case_index": i, "case_id": str(case_cfg.get("id", "")), **compact_exception(exc)})
    else:
        module_name = str(cfg.get("d17_g6f_module", "gv1.d17_g.g6f_selected_cycle_infer"))
        g6f = importlib.import_module(module_name)
        split_manifest = Path(str(cfg.get("split_manifest", "")))
        semantics_csv = Path(str(cfg.get("semantics_csv", "")))
        internal_manifest_raw = str(cfg.get("internal_heldout_manifest", "")).strip()
        internal_manifest = Path(internal_manifest_raw) if internal_manifest_raw else None
        candidate_dir = Path(str(cfg.get("candidate_dir", "")))
        candidate_summary = Path(str(cfg.get("candidate_summary", ""))) if str(cfg.get("candidate_summary", "")).strip() else None
        checkpoint = Path(str(cfg.get("checkpoint", ""))) if str(cfg.get("checkpoint", "")).strip() else None
        required_paths = {
            "split_manifest": split_manifest,
            "semantics_csv": semantics_csv,
            "internal_heldout_manifest": internal_manifest,
            "candidate_dir": candidate_dir,
        }
        if checkpoint is not None:
            required_paths["checkpoint"] = checkpoint
        for label, required_path in required_paths.items():
            if required_path is None or not required_path.exists():
                raise FileNotFoundError(f"Required D17 dense-export path is missing: {label}={required_path}")
        records, _ = g6f.load_split_records(split_manifest)
        semantics_map = g6f.load_semantics_map(semantics_csv)
        ih_uids = _internal_heldout_uids(internal_manifest)
        ckpt, checkpoint_path, _ = g6f.load_candidate_checkpoint(candidate_dir, candidate_summary, checkpoint)
        device = g6f.device_from_arg(str(cfg.get("device", "auto")))
        model = g6f.build_model_from_checkpoint(ckpt, device)
        feature_names = list(ckpt.get("feature_names") or [])
        protocol_vocab, branch_vocab = g6f.parse_vocabs_from_checkpoint_feature_names(
            feature_names, int(ckpt.get("local_input_dim"))
        )
        target_slices = {str(k): (int(v[0]), int(v[1])) for k, v in dict(ckpt.get("target_slices") or {}).items()}
        metric_targets = [key for key in STATE_KEYS if key in target_slices]
        missing_targets = [key for key in STATE_KEYS if key not in metric_targets]
        if missing_targets:
            raise ValueError(f"Checkpoint target_slices are missing required D18 states: {missing_targets}")

        for i, case_cfg in enumerate(case_cfgs):
            case_id = str(case_cfg.get("id", f"case_{i:02d}"))
            try:
                record = g6f.find_record(records, str(case_cfg.get("batch")), str(case_cfg.get("battery")))
                uid = str(record.get("canonical_cell_uid") or record.get("cell_uid") or case_id)
                split = "internal_heldout" if _norm(uid) in ih_uids else _split_name(record.get("split"))
                expected = _split_name(case_cfg.get("expected_split"))
                if expected and split != expected:
                    raise ValueError(f"Split mismatch for {uid}: expected {expected}, got {split}")
                if split in BLOCKED_SPLITS:
                    raise ValueError(f"Blocked split selected for dense diagnostic: {uid} split={split}")
                sem_row = g6f.semantics_for_record(record, semantics_map)
                if not sem_row:
                    raise ValueError(f"No G0 semantic-branch row found for {uid}")
                soft_path = Path(str(record.get("softlabel_npz") or sem_row.get("softlabel_npz") or ""))
                selected_cycles, groups = choose_cycles(soft_path, case_cfg, int(cfg.get("cycles_per_position", 3)))
                cycle_spec = ",".join(str(x) for x in selected_cycles)
                data = g6f.load_selected_cycle_data(
                    record=record,
                    sem_row=sem_row,
                    cycles=cycle_spec,
                    protocol_vocab=protocol_vocab,
                    branch_vocab=branch_vocab,
                    metric_targets=metric_targets,
                    max_points_per_cycle=int(cfg.get("max_points_per_cycle", 0)),
                    prefer_replay_observed=bool(cfg.get("prefer_replay_observed", True)),
                )
                expected_dim = int(np.asarray(ckpt["x_mean"]).size)
                if int(data["X"].shape[1]) != expected_dim:
                    raise ValueError(f"Feature dimension mismatch: built={data['X'].shape[1]}, checkpoint={expected_dim}")
                pred = g6f.predict_array(
                    model,
                    data["X"],
                    ckpt,
                    device,
                    batch_size=int(cfg.get("prediction_batch_size", 8192)),
                )
                pred_by_target = g6f.slice_prediction_by_targets(pred, target_slices, metric_targets)
                output_path = cases_dir / f"{i:02d}_{case_id}.npz"
                row = _save_real_case(
                    case_cfg=case_cfg,
                    record=record,
                    sem_row=sem_row,
                    data=data,
                    pred_by_target=pred_by_target,
                    split=split,
                    cycle_groups=groups,
                    output_path=output_path,
                    checkpoint_path=checkpoint_path,
                )
                min_points = int(cfg.get("dense_min_time_points", 768))
                if int(row["n_time"]) < min_points:
                    raise ValueError(f"Dense case has only {row['n_time']} points; minimum is {min_points}")
                manifest_rows.append(row)
                case_files.append(output_path)
            except Exception as exc:
                failures.append({
                    "case_index": i,
                    "case_id": case_id,
                    "batch": str(case_cfg.get("batch", "")),
                    "battery": str(case_cfg.get("battery", "")),
                    **compact_exception(exc),
                })

    write_csv(manifest_rows, out / "D18_S1_DENSE_CASEPACK_MANIFEST.csv", fieldnames=None if manifest_rows else [
        "case_id", "canonical_cell_uid", "split", "role", "protocol", "semantic_branch", "branch_family",
        "selected_cycles", "cycle_count", "n_time", "prediction_npz", "prediction_sha256", "status",
    ])
    write_csv(failures, out / "D18_S1_DENSE_CASEPACK_FAILURES.csv", fieldnames=None if failures else [
        "case_index", "case_id", "batch", "battery", "type", "message",
    ])
    expected_count = len(case_cfgs)
    blocked = [r for r in manifest_rows if _split_name(r.get("split")) in BLOCKED_SPLITS]
    status = "PASS" if expected_count > 0 and len(manifest_rows) == expected_count and not failures and not blocked else "FAIL"
    summary = {
        "stage": "D18-S1-DENSE-CASEPACK-FIX",
        "created_at_utc": utc_now_iso(),
        "status": status,
        "training_launched": False,
        "synthetic_fixture_mode": synthetic,
        "expected_case_count": expected_count,
        "actual_case_count": len(manifest_rows),
        "failure_count": len(failures),
        "blocked_case_count": len(blocked),
        "casepack_dir": str(cases_dir),
        "case_ids": [str(r.get("case_id")) for r in manifest_rows],
        "splits": sorted(set(str(r.get("split")) for r in manifest_rows)),
        "protocols": sorted(set(str(r.get("protocol")) for r in manifest_rows)),
        "branch_families": sorted(set(str(r.get("branch_family")) for r in manifest_rows)),
        "failures": failures,
        "go_to_s2": False,
    }
    dump_json(summary, out / "D18_S1_DENSE_CASEPACK_SUMMARY.json")
    return DenseCasepackResult(status, out, case_files, manifest_rows, failures, summary)
