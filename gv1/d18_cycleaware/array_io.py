from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import compact_exception, expand_candidate_paths, resolve_config_path
from .schema import (
    CYCLE_ALIASES,
    CURRENT_ALIASES,
    META_ALIASES,
    PRED_ALIASES,
    RADIAL_GRID_ALIASES,
    STATE_KEYS,
    STEP_ALIASES,
    TEMP_ALIASES,
    TIME_ALIASES,
    TRUE_ALIASES,
    VOLTAGE_ALIASES,
    ArrayCase,
    as_1d_any,
    as_1d_numeric,
    extract_meta,
    find_key,
    get_radial_grid,
    get_state_arrays,
    get_time,
    infer_time_length,
    linear_align,
    load_npz_selected,
    nearest_align,
    npz_keys,
    orient_time_first,
    scalar_string,
)


_UID_RE = re.compile(r"(batch[-_ ]?\d+[^\\/]*?battery[-_ ]?\d+)", re.IGNORECASE)
_CYCLE_RANGE_RE = re.compile(r"cycles?[-_ ]?(\d+)[-_ ]?(?:to|through|-)?[-_ ]?(\d+)?", re.IGNORECASE)


@dataclass
class DiscoveryResult:
    cases: list[ArrayCase]
    inventory_rows: list[dict[str, Any]]
    warnings: list[dict[str, Any]]


def normalize_uid(value: str) -> str:
    text = value.strip().lower().replace("\\", "/")
    return re.sub(r"[^a-z0-9]+", "", text)


def infer_uid_from_path(path: str | Path) -> str:
    text = str(path).replace("\\", "/")
    match = _UID_RE.search(text)
    if match:
        return match.group(1).replace(" ", "_")
    return Path(path).stem


def safe_case_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return text[:180] or "d18_case"


def _normalize_split_name(value: str | None) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "val": "validation",
        "valid": "validation",
        "internalheldout": "internal_heldout",
        "heldout": "internal_heldout",
        "frozen": "frozen_test",
        "frozentest": "frozen_test",
        "probe": "flagged_probe",
        "flagged": "flagged_probe",
    }
    return aliases.get(text, text)


def _split_records_from_json(data: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    split_keys = {
        "train", "validation", "val", "internal_heldout", "heldout",
        "frozen_test", "test", "flagged_probe", "flagged", "probe",
    }
    id_keys = {
        "canonical_cell_uid", "cell_uid", "profile_id", "battery", "softlabel_npz",
        "softlabel_path", "solution_softlabels_npz", "replay_npz", "source_softlabel_npz",
    }

    normalized_split_keys = {_normalize_split_name(k) for k in split_keys}

    def walk(node: Any, inherited_split: str = "") -> None:
        if isinstance(node, Mapping):
            explicit_split = _normalize_split_name(str(node.get("split", inherited_split)))
            is_record = any(k in node for k in id_keys)
            if is_record:
                row = dict(node)
                if explicit_split and not row.get("split"):
                    row["split"] = explicit_split
                records.append(row)
            for key, value in node.items():
                key_norm = _normalize_split_name(str(key))
                child_split = key_norm if key_norm in normalized_split_keys else explicit_split
                # Once a mapping is recognized as a record, its scalar fields are
                # attributes, not additional UID records. Continue only into nested
                # containers so rich rows cannot be overwritten by leaf strings.
                if is_record and not isinstance(value, Mapping) and not (
                    isinstance(value, Sequence) and not isinstance(value, (str, bytes))
                ):
                    continue
                walk(value, child_split)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for value in node:
                walk(value, inherited_split)
        elif isinstance(node, (str, bytes)) and inherited_split:
            text = str(node).strip()
            if text:
                records.append({"canonical_cell_uid": text, "split": inherited_split})

    walk(data)
    return records


def load_split_index(split_manifest_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if split_manifest_path is None:
        return {}
    p = Path(split_manifest_path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
    records = _split_records_from_json(data)
    out: dict[str, dict[str, Any]] = {}
    index_fields = (
        "canonical_cell_uid", "cell_uid", "profile_id", "battery", "softlabel_npz",
        "softlabel_path", "solution_softlabels_npz", "replay_npz", "source_softlabel_npz",
    )
    def record_score(row: Mapping[str, Any]) -> tuple[int, int]:
        path_fields = ("softlabel_npz", "softlabel_path", "solution_softlabels_npz", "source_softlabel_npz")
        return (sum(bool(str(row.get(k, "")).strip()) for k in path_fields), len(row))

    for record in records:
        row = dict(record)
        row["split"] = _normalize_split_name(str(row.get("split", "UNKNOWN"))) or "UNKNOWN"
        for key in index_fields:
            value = str(row.get(key, "")).strip()
            if not value:
                continue
            aliases = (normalize_uid(value), normalize_uid(Path(value).stem), normalize_uid(Path(value).parent.name))
            for alias in aliases:
                if not alias:
                    continue
                previous = out.get(alias)
                if previous is None or record_score(row) > record_score(previous):
                    out[alias] = row
    return out


def _record_for_uid(split_index: Mapping[str, dict[str, Any]], *values: str) -> dict[str, Any] | None:
    for value in values:
        if not value:
            continue
        key = normalize_uid(value)
        if key in split_index:
            return split_index[key]
        for idx_key, record in split_index.items():
            if key and (key in idx_key or idx_key in key):
                return record
    return None


def prediction_key_set(keys: Sequence[str]) -> set[str]:
    keyset = set(keys)
    return {state for state in STATE_KEYS if any(alias in keyset for alias in PRED_ALIASES[state])}


def paired_true_key_set(keys: Sequence[str]) -> set[str]:
    keyset = set(keys)
    return {state for state in STATE_KEYS if any(alias in keyset for alias in TRUE_ALIASES[state])}


def discover_prediction_files(
    roots: Sequence[str],
    globs: Sequence[str],
    config: Mapping[str, Any],
    project_root: str | Path,
    max_files: int,
) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for raw_root in roots:
        root = resolve_config_path(raw_root, config, project_root)
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() == ".npz":
            candidates = [root]
        else:
            candidates = []
            for pattern in globs:
                candidates.extend(root.glob(pattern))
        for path in candidates:
            if not path.is_file() or path.suffix.lower() != ".npz":
                continue
            rp = path.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            found.append(rp)
            if len(found) >= max_files:
                return sorted(found)
    return sorted(found)


def _selected_npz_keys(keys: Sequence[str], include_true: bool = True) -> list[str]:
    selected: set[str] = set()
    for state in STATE_KEYS:
        selected.update(alias for alias in PRED_ALIASES[state] if alias in keys)
        if include_true:
            selected.update(alias for alias in TRUE_ALIASES[state] if alias in keys)
    for aliases in (TIME_ALIASES, CYCLE_ALIASES, CURRENT_ALIASES, VOLTAGE_ALIASES, TEMP_ALIASES, STEP_ALIASES):
        selected.update(alias for alias in aliases if alias in keys)
    for aliases in RADIAL_GRID_ALIASES.values():
        selected.update(alias for alias in aliases if alias in keys)
    for aliases in META_ALIASES.values():
        selected.update(alias for alias in aliases if alias in keys)
    selected.update(key for key in keys if key in {"source_info", "cycle_from", "cycle_to", "selected_cycles"})
    return sorted(selected)


def _truth_selected_keys(keys: Sequence[str]) -> list[str]:
    selected: set[str] = set()
    for state in STATE_KEYS:
        selected.add(state) if state in keys else None
        selected.update(alias for alias in TRUE_ALIASES[state] if alias in keys)
    for aliases in (TIME_ALIASES, CYCLE_ALIASES, CURRENT_ALIASES, VOLTAGE_ALIASES, TEMP_ALIASES, STEP_ALIASES):
        selected.update(alias for alias in aliases if alias in keys)
    for aliases in RADIAL_GRID_ALIASES.values():
        selected.update(alias for alias in aliases if alias in keys)
    for aliases in META_ALIASES.values():
        selected.update(alias for alias in aliases if alias in keys)
    return sorted(selected)


def _explicit_truth_path(pred_mapping: Mapping[str, Any], prediction_path: Path) -> Path | None:
    key = find_key(pred_mapping, META_ALIASES["source_softlabel_npz"])
    if key is None:
        return None
    raw = scalar_string(pred_mapping[key]).strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = prediction_path.parent / p
    return p.resolve()


def _manifest_truth_path(record: Mapping[str, Any] | None) -> Path | None:
    if not record:
        return None
    for key in ("softlabel_npz", "solution_softlabels_npz", "softlabel_path", "source_softlabel_npz"):
        raw = str(record.get(key, "")).strip()
        if raw:
            return Path(raw).resolve()
    return None


def _load_truth_mapping(path: Path, max_bytes: int) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size > max_bytes:
        return {}
    keys = npz_keys(path)
    return load_npz_selected(path, _truth_selected_keys(keys))


def _aligned_optional_numeric(
    pred_map: Mapping[str, Any],
    truth_map: Mapping[str, Any],
    aliases: Sequence[str],
    pred_t: np.ndarray,
    truth_t: np.ndarray | None,
) -> np.ndarray | None:
    key = find_key(pred_map, aliases)
    if key is not None:
        try:
            arr = as_1d_numeric(pred_map[key], key)
            if arr.size == pred_t.size:
                return arr
        except Exception:
            pass
    key = find_key(truth_map, aliases)
    if key is not None:
        try:
            arr = as_1d_numeric(truth_map[key], key)
            if arr.size == pred_t.size:
                return arr
            if truth_t is not None and arr.size == truth_t.size:
                return linear_align(truth_t, arr[:, None], pred_t)[:, 0]
        except Exception:
            pass
    return None


def _aligned_optional_any(
    pred_map: Mapping[str, Any],
    truth_map: Mapping[str, Any],
    aliases: Sequence[str],
    pred_t: np.ndarray,
    truth_t: np.ndarray | None,
) -> np.ndarray | None:
    key = find_key(pred_map, aliases)
    if key is not None:
        arr = as_1d_any(pred_map[key])
        if arr.size == pred_t.size:
            return arr
    key = find_key(truth_map, aliases)
    if key is not None:
        arr = as_1d_any(truth_map[key])
        if arr.size == pred_t.size:
            return arr
        if truth_t is not None and arr.size == truth_t.size:
            return nearest_align(truth_t, arr, pred_t)
    return None


def _extract_cycle_ranges(config_s1: Mapping[str, Any]) -> list[tuple[int, int]]:
    raw = config_s1.get("cycle_ranges", [])
    ranges: list[tuple[int, int]] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
                ranges.append((int(item[0]), int(item[1])))
    return ranges


def _filter_case_by_cycles(case: ArrayCase, ranges: Sequence[tuple[int, int]]) -> ArrayCase | None:
    if not ranges:
        return case
    if case.cycle_id is None:
        return None
    cycle = np.asarray(case.cycle_id).reshape(-1)
    mask = np.zeros(cycle.size, dtype=bool)
    for start, stop in ranges:
        lo, hi = min(start, stop), max(start, stop)
        with np.errstate(invalid="ignore"):
            mask |= (cycle.astype(float) >= lo) & (cycle.astype(float) <= hi)
    if not np.any(mask):
        return None
    idx = np.flatnonzero(mask)
    # Preserve all points inside each selected cycle, even if they are non-contiguous in unusual files.
    return _slice_case(case, idx)


def _slice_case(case: ArrayCase, idx: np.ndarray) -> ArrayCase:
    def sel(value: np.ndarray | None) -> np.ndarray | None:
        return None if value is None else np.asarray(value)[idx]

    return ArrayCase(
        case_id=case.case_id,
        prediction_path=case.prediction_path,
        truth_path=case.truth_path,
        canonical_cell_uid=case.canonical_cell_uid,
        cell_uid=case.cell_uid,
        protocol=case.protocol,
        branch=case.branch,
        split=case.split,
        time_s=np.asarray(case.time_s)[idx],
        cycle_id=sel(case.cycle_id),
        current_A=sel(case.current_A),
        voltage_V=sel(case.voltage_V),
        temperature_C=sel(case.temperature_C),
        step_type=sel(case.step_type),
        radial_grid_a=case.radial_grid_a,
        radial_grid_c=case.radial_grid_c,
        pred={k: np.asarray(v)[idx] for k, v in case.pred.items()},
        true={k: np.asarray(v)[idx] for k, v in case.true.items()},
        metadata=dict(case.metadata),
    )


def load_array_case(
    prediction_path: str | Path,
    split_index: Mapping[str, dict[str, Any]],
    max_truth_file_bytes: int,
) -> ArrayCase:
    pred_path = Path(prediction_path).resolve()
    pred_keys = npz_keys(pred_path)
    pred_states = prediction_key_set(pred_keys)
    if not pred_states:
        raise ValueError("NPZ has no recognized prediction state arrays")
    pred_map = load_npz_selected(pred_path, _selected_npz_keys(pred_keys, include_true=True))
    n_pred = infer_time_length(pred_map, prefer_pred=True)
    pred_t = get_time(pred_map, n_pred)
    pred = get_state_arrays(pred_map, "pred", n_pred)
    paired_true = get_state_arrays(pred_map, "true", n_pred)

    inferred_uid = infer_uid_from_path(pred_path)
    canonical = extract_meta(pred_map, "canonical_cell_uid", inferred_uid)
    cell_uid = extract_meta(pred_map, "cell_uid", canonical)
    record = _record_for_uid(split_index, canonical, cell_uid, inferred_uid, str(pred_path))
    path_lower = str(pred_path).lower().replace("-", "_")
    inferred_split = str(record.get("split", "UNKNOWN")) if record else "UNKNOWN"
    if inferred_split.upper() == "UNKNOWN":
        if "flagged_probe" in path_lower or "flagged" in path_lower:
            inferred_split = "flagged_probe"
        elif "frozen_test" in path_lower or "frozentest" in path_lower:
            inferred_split = "frozen_test"
        elif re.search(r"(^|[\\/_])test([\\/_]|$)", path_lower):
            inferred_split = "test"
    split = _normalize_split_name(extract_meta(pred_map, "split", inferred_split)) or "unknown"
    protocol_default = str(record.get("protocol", "UNKNOWN")) if record else "UNKNOWN"
    protocol = extract_meta(pred_map, "protocol", protocol_default)
    branch_default = "UNKNOWN"
    if record:
        branch_default = str(record.get("semantic_branch") or record.get("generator_branch") or record.get("branch") or "UNKNOWN")
    branch = extract_meta(pred_map, "branch", branch_default)

    truth_path: Path | None = None
    truth_map: dict[str, Any] = {}
    truth_t: np.ndarray | None = None
    true: dict[str, np.ndarray] = dict(paired_true)
    if len(true) < len(pred):
        truth_path = _explicit_truth_path(pred_map, pred_path) or _manifest_truth_path(record)
        if truth_path is not None and truth_path.exists() and truth_path.resolve() != pred_path:
            truth_map = _load_truth_mapping(truth_path, max_truth_file_bytes)
            if truth_map:
                n_truth = infer_time_length(truth_map, prefer_pred=False)
                truth_t = get_time(truth_map, n_truth)
                raw_true = get_state_arrays(truth_map, "raw_true", n_truth)
                for state, values in raw_true.items():
                    if state not in true:
                        true[state] = linear_align(truth_t, values, pred_t)
    if not true:
        raise ValueError("No paired truth arrays were found in prediction NPZ or resolved soft-label NPZ")

    cycle = _aligned_optional_any(pred_map, truth_map, CYCLE_ALIASES, pred_t, truth_t)
    current = _aligned_optional_numeric(pred_map, truth_map, CURRENT_ALIASES, pred_t, truth_t)
    voltage = _aligned_optional_numeric(pred_map, truth_map, VOLTAGE_ALIASES, pred_t, truth_t)
    temp = _aligned_optional_numeric(pred_map, truth_map, TEMP_ALIASES, pred_t, truth_t)
    step = _aligned_optional_any(pred_map, truth_map, STEP_ALIASES, pred_t, truth_t)

    radial_source = pred_map if pred_map else truth_map
    nra = pred.get("cs_a", pred.get("theta_a", np.empty((n_pred, 0)))).shape[1] if any(s in pred for s in ("cs_a", "theta_a")) else 0
    nrc = pred.get("cs_c", pred.get("theta_c", np.empty((n_pred, 0)))).shape[1] if any(s in pred for s in ("cs_c", "theta_c")) else 0
    r_a = get_radial_grid(radial_source if find_key(radial_source, RADIAL_GRID_ALIASES["a"]) else truth_map, "a", nra) if nra else None
    r_c = get_radial_grid(radial_source if find_key(radial_source, RADIAL_GRID_ALIASES["c"]) else truth_map, "c", nrc) if nrc else None

    common_states = [state for state in STATE_KEYS if state in pred and state in true]
    if not common_states:
        raise ValueError("No states have both prediction and truth arrays")
    # Trim to a common time length if an unusual file has inconsistent state arrays.
    n_common = min([n_pred] + [pred[s].shape[0] for s in common_states] + [true[s].shape[0] for s in common_states])
    embedded_case_id = extract_meta(pred_map, "case_id", "")
    case_name = safe_case_id(embedded_case_id or f"{canonical}__{pred_path.stem}")
    extra_meta = {
        "branch_family": extract_meta(pred_map, "branch_family", ""),
        "case_role": extract_meta(pred_map, "case_role", ""),
        "casepack_version": extract_meta(pred_map, "casepack_version", ""),
    }
    case = ArrayCase(
        case_id=case_name,
        prediction_path=str(pred_path),
        truth_path=str(truth_path) if truth_path is not None else (str(pred_path) if paired_true else None),
        canonical_cell_uid=canonical,
        cell_uid=cell_uid,
        protocol=protocol,
        branch=branch,
        split=split,
        time_s=pred_t[:n_common],
        cycle_id=None if cycle is None else cycle[:n_common],
        current_A=None if current is None else current[:n_common],
        voltage_V=None if voltage is None else voltage[:n_common],
        temperature_C=None if temp is None else temp[:n_common],
        step_type=None if step is None else step[:n_common],
        radial_grid_a=r_a,
        radial_grid_c=r_c,
        pred={s: pred[s][:n_common] for s in common_states},
        true={s: true[s][:n_common] for s in common_states},
        metadata={
            "prediction_keys": pred_keys,
            "paired_truth_states": sorted(paired_true),
            "resolved_truth_path": str(truth_path) if truth_path is not None else "",
            "manifest_record": dict(record) if record else None,
            **extra_meta,
        },
    )
    return case


def discover_array_cases(
    config: Mapping[str, Any],
    project_root: str | Path,
) -> DiscoveryResult:
    s1 = config.get("s1", {})
    if not isinstance(s1, Mapping):
        raise ValueError("config.s1 must be an object")
    paths = config.get("paths", {}) if isinstance(config.get("paths"), Mapping) else {}
    split_manifest_raw = str(paths.get("d17_split_manifest", ""))
    split_manifest = resolve_config_path(split_manifest_raw, config, project_root) if split_manifest_raw else None
    split_index = load_split_index(split_manifest)

    roots = [str(x) for x in s1.get("prediction_roots", [])]
    globs = [str(x) for x in s1.get("prediction_globs", ["**/*.npz"])]
    max_files = int(s1.get("max_candidate_files", 5000))
    files = discover_prediction_files(roots, globs, config, project_root, max_files)
    allowed_splits = {str(x).lower() for x in s1.get("allowed_splits", ["train", "validation", "internal_heldout", "unknown"])}
    blocked_splits = {str(x).lower() for x in s1.get("blocked_splits", ["frozen_test", "flagged_probe", "test"])}
    include_uids = [normalize_uid(str(x)) for x in s1.get("include_uids", [])]
    exclude_uids = [normalize_uid(str(x)) for x in s1.get("exclude_uids", [])]
    min_points = int(s1.get("min_time_points", 128))
    dense_min = int(s1.get("dense_min_time_points", 768))
    require_dense = bool(s1.get("require_dense", True))
    max_cases = int(s1.get("max_cases", 24))
    max_truth_bytes = int(s1.get("max_truth_file_bytes", 2 * 1024**3))
    cycle_ranges = _extract_cycle_ranges(s1)

    cases: list[ArrayCase] = []
    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for path in files:
        row: dict[str, Any] = {"prediction_path": str(path), "status": "PENDING"}
        try:
            keys = npz_keys(path)
            pred_states = sorted(prediction_key_set(keys))
            true_states = sorted(paired_true_key_set(keys))
            row.update({"pred_states": pred_states, "paired_true_states": true_states, "size_bytes": path.stat().st_size})
            if not pred_states:
                row["status"] = "SKIP_NOT_PREDICTION_NPZ"
                rows.append(row)
                continue
            case = load_array_case(path, split_index, max_truth_bytes)
            uid_norm = normalize_uid(case.canonical_cell_uid)
            if include_uids and not any(token in uid_norm or uid_norm in token for token in include_uids):
                row["status"] = "SKIP_UID_NOT_INCLUDED"
                rows.append(row)
                continue
            if any(token in uid_norm or uid_norm in token for token in exclude_uids):
                row["status"] = "SKIP_UID_EXCLUDED"
                rows.append(row)
                continue
            split_lower = case.split.lower()
            if split_lower in blocked_splits:
                row["status"] = "SKIP_BLOCKED_SPLIT"
                row["split"] = case.split
                rows.append(row)
                continue
            if allowed_splits and split_lower not in allowed_splits:
                row["status"] = "SKIP_SPLIT_NOT_ALLOWED"
                row["split"] = case.split
                rows.append(row)
                continue
            case = _filter_case_by_cycles(case, cycle_ranges)
            if case is None:
                row["status"] = "SKIP_CYCLE_RANGE_NO_MATCH"
                rows.append(row)
                continue
            if case.n_time < min_points:
                row["status"] = "SKIP_TOO_FEW_POINTS"
                row["n_time"] = case.n_time
                rows.append(row)
                continue
            if require_dense and case.n_time < dense_min:
                row["status"] = "SKIP_NOT_DENSE"
                row["n_time"] = case.n_time
                rows.append(row)
                continue
            row.update(
                {
                    "status": "SELECTED",
                    "case_id": case.case_id,
                    "canonical_cell_uid": case.canonical_cell_uid,
                    "split": case.split,
                    "protocol": case.protocol,
                    "branch": case.branch,
                    "n_time": case.n_time,
                    "available_states": case.available_states,
                    "truth_path": case.truth_path or "",
                }
            )
            cases.append(case)
            rows.append(row)
            if len(cases) >= max_cases:
                break
        except Exception as exc:
            row["status"] = "ERROR"
            row.update(compact_exception(exc))
            rows.append(row)
            warnings.append({"prediction_path": str(path), **compact_exception(exc)})
    return DiscoveryResult(cases=cases, inventory_rows=rows, warnings=warnings)
