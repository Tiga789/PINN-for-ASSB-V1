from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from .reader_base import ReadOptions, STANDARD_COLUMNS
from .unit_normalizer import (
    normalize_capacity_to_ah,
    normalize_current_to_ampere,
    normalize_current_to_ampere as _norm_current,
    normalize_energy_to_wh,
    normalize_sign_current,
    normalize_temperature_to_celsius,
    normalize_time_to_seconds,
    normalize_voltage_to_volt,
)


def _clean_name(name: Any) -> str:
    s = str(name).strip()
    s = s.replace("\ufeff", "")
    s = re.sub(r"[\s\-\/\\()\[\]{}]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# Candidate aliases. The mapper is deliberately broad because .mat exports vary.
ALIASES: dict[str, list[str]] = {
    "time_s": [
        "time", "time_s", "times", "test_time", "relative_time", "relative_time_s",
        "relative_time_min", "time_min", "t", "t_s", "sec", "seconds", "date_time",
    ],
    "current_A": ["current", "current_a", "i", "i_a", "curr", "current_ma", "charge_current", "discharge_current"],
    "voltage_V": ["voltage", "voltage_v", "v", "u", "terminal_voltage", "voltage_mv"],
    "capacity_Ah": ["capacity", "capacity_ah", "q", "q_ah", "capacity_mah", "charge_capacity", "discharge_capacity"],
    "energy_Wh": ["energy", "energy_wh", "power_wh", "watt_hour", "wh"],
    "temperature_C": ["temperature", "temperature_c", "temp", "temp_c", "temperature_k", "ambient_temperature"],
    "cycle_id": ["cycle", "cycle_id", "cycle_index", "cycle_number", "cyc", "cycle_no"],
    "step_id": ["step", "step_id", "step_index", "step_number", "stage", "stage_id"],
    "step_type": ["step_type", "state", "status", "mode", "description", "desc", "operation", "procedure"],
    "dataset_id": ["dataset_id", "dataset"],
    "batch_id": ["batch_id", "batch"],
    "battery_id": ["battery_id", "cell_id", "cell", "battery", "barcode"],
}

_ALIAS_LOOKUP = {alias: std for std, aliases in ALIASES.items() for alias in aliases}


def infer_standard_column(raw_name: str, explicit_map: Mapping[str, str] | None = None) -> Optional[str]:
    cleaned = _clean_name(raw_name)
    low = cleaned.lower()
    # For XJTU/MATLAB battery files, system_time can jump backwards across subrecords
    # after the public data have been stitched. Keep it only as raw__system_time;
    # prefer relative_time_min plus raw__mat_subrecord_index for global time reconstruction.
    if low in {"system_time", "timestamp", "date_time"}:
        return None
    if explicit_map:
        for k, v in explicit_map.items():
            if _clean_name(k).lower() == low:
                return v
    if low in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[low]
    # Fuzzy fallbacks.
    if "volt" in low and not "delta" in low:
        return "voltage_V"
    if ("current" in low or low in {"i", "curr"}) and "capacity" not in low:
        return "current_A"
    if "temperature" in low or low.startswith("temp"):
        return "temperature_C"
    if "capacity" in low or low in {"q", "cap"}:
        return "capacity_Ah"
    if "energy" in low or low.endswith("wh"):
        return "energy_Wh"
    if "cycle" in low:
        return "cycle_id"
    if "step" in low or "stage" in low:
        return "step_id" if "type" not in low else "step_type"
    if "description" in low or "mode" in low or "status" in low:
        return "step_type"
    # Do not map system_time to time_s if a numeric time column also exists; handled by scoring/order.
    if low in {"time", "t", "relative_time", "relative_time_s", "relative_time_min", "test_time"}:
        return "time_s"
    return None


def _unit_hint(options: ReadOptions, raw_name: str, std_name: str) -> Optional[str]:
    for key in (raw_name, _clean_name(raw_name), std_name):
        if key in options.unit_map:
            return options.unit_map[key]
    return None


def _convert_column(raw: pd.Series, raw_name: str, std_name: str, options: ReadOptions) -> pd.Series:
    unit = _unit_hint(options, raw_name, std_name)
    if std_name == "time_s":
        return normalize_time_to_seconds(raw, unit, raw_name)
    if std_name == "current_A":
        return normalize_current_to_ampere(raw, unit, raw_name)
    if std_name == "voltage_V":
        return normalize_voltage_to_volt(raw, unit, raw_name)
    if std_name == "capacity_Ah":
        return normalize_capacity_to_ah(raw, unit, raw_name)
    if std_name == "energy_Wh":
        return normalize_energy_to_wh(raw, unit, raw_name)
    if std_name == "temperature_C":
        return normalize_temperature_to_celsius(raw, unit, raw_name)
    if std_name in {"cycle_id", "step_id"}:
        return pd.to_numeric(raw, errors="coerce").astype("Int64")
    if std_name in {"dataset_id", "batch_id", "battery_id", "step_type"}:
        return raw.astype("string")
    return raw


def _classify_step_type(current: pd.Series, threshold_A: float = 1e-9) -> pd.Series:
    cur = pd.to_numeric(current, errors="coerce")
    out = np.full(len(cur), "unknown", dtype=object)
    out[cur > threshold_A] = "charge"
    out[cur < -threshold_A] = "discharge"
    out[cur.abs() <= threshold_A] = "rest"
    return pd.Series(out, index=current.index, dtype="string")



def _is_monotonic_non_decreasing(series: pd.Series) -> bool:
    t = pd.to_numeric(series, errors="coerce")
    if t.notna().sum() <= 1:
        return True
    dt = t.dropna().diff().dropna()
    return not bool((dt < 0).any())


def _positive_dt_guess(*series_list: pd.Series) -> float:
    """Return a conservative positive sampling interval guess in seconds."""
    candidates: list[float] = []
    for series in series_list:
        vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) <= 2:
            continue
        diffs = np.diff(vals)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if len(diffs):
            # Use the median but clamp absurdly small values.
            candidates.append(float(np.nanmedian(diffs)))
    for c in candidates:
        if np.isfinite(c) and c > 0:
            return max(c, 1e-6)
    return 1.0


def _local_monotonic_elapsed_seconds(values: pd.Series) -> pd.Series | None:
    """Convert a local time-like series to elapsed seconds if it is usable."""
    vals = pd.to_numeric(values, errors="coerce")
    if vals.notna().sum() <= 1:
        return None
    # Work with the valid values only for monotonicity, then subtract the first valid value.
    valid = vals.dropna()
    diffs = valid.diff().dropna()
    if (diffs < 0).any():
        return None
    first = valid.iloc[0]
    elapsed = vals - first
    return elapsed.astype(float)


def _repair_nonmonotonic_time_axis(out: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    """Repair non-monotonic ``time_s`` for concatenated MATLAB/XJTU-style records.

    Some public battery datasets store one .mat file as a list of subrecords/cycles.
    When we concatenate all subrecords, timestamp-like columns such as ``system_time``
    may jump backwards due to acquisition pauses or stitching, while the row order still
    represents the experimental sequence. For measured-current replay, we need a single
    monotonically increasing ``t_global_s``. This helper rebuilds ``time_s`` by preserving
    subrecord order and using local elapsed time within each contiguous subrecord.
    """
    if "time_s" not in out.columns or _is_monotonic_non_decreasing(out["time_s"]):
        return out

    # Case 1: .mat reader provided subrecord IDs. Rebuild a global axis by contiguous
    # subrecord segments, using local system_time if valid, otherwise relative_time_min,
    # otherwise row index with a 1 s fallback.
    if "raw__mat_subrecord_index" in out.columns:
        idx = pd.Series(out["raw__mat_subrecord_index"]).reset_index(drop=True)
        segment_id = idx.ne(idx.shift()).cumsum()
        repaired = np.full(len(out), np.nan, dtype=float)
        offset = 0.0
        global_dt_guess = _positive_dt_guess(pd.to_numeric(out.get("time_s"), errors="coerce"))

        rel_raw = out.get("raw__relative_time_min")
        rel_sec_all = None
        if rel_raw is not None:
            # The XJTU .mat files label this field in minutes.
            rel_sec_all = pd.to_numeric(rel_raw, errors="coerce") * 60.0
            global_dt_guess = _positive_dt_guess(rel_sec_all, pd.to_numeric(out.get("time_s"), errors="coerce"))

        for _, loc in segment_id.groupby(segment_id).groups.items():
            loc = list(loc)
            if not loc:
                continue
            orig_local = _local_monotonic_elapsed_seconds(pd.Series(out["time_s"].iloc[loc]).reset_index(drop=True))
            rel_local = None
            if rel_sec_all is not None:
                rel_local = _local_monotonic_elapsed_seconds(pd.Series(rel_sec_all.iloc[loc]).reset_index(drop=True))

            if orig_local is not None and orig_local.notna().sum() > 1:
                local = orig_local
            elif rel_local is not None and rel_local.notna().sum() > 1:
                local = rel_local
            else:
                local_dt = global_dt_guess if np.isfinite(global_dt_guess) and global_dt_guess > 0 else 1.0
                local = pd.Series(np.arange(len(loc), dtype=float) * local_dt)

            local_vals = pd.to_numeric(local, errors="coerce").to_numpy(dtype=float)
            if len(local_vals) and not np.isfinite(local_vals[0]):
                # Fill sparse NaNs by row index if the first value is missing.
                local_dt = global_dt_guess if np.isfinite(global_dt_guess) and global_dt_guess > 0 else 1.0
                local_vals = np.arange(len(loc), dtype=float) * local_dt
            # Forward fill any internal NaNs and enforce non-negative elapsed time.
            local_series = pd.Series(local_vals).ffill().bfill().fillna(0.0)
            local_vals = local_series.to_numpy(dtype=float)
            local_vals = local_vals - float(local_vals[0])
            local_vals = np.maximum.accumulate(local_vals)
            repaired[loc] = offset + local_vals

            positive = np.diff(local_vals)
            positive = positive[np.isfinite(positive) & (positive > 0)]
            step = float(np.nanmedian(positive)) if len(positive) else global_dt_guess
            if not np.isfinite(step) or step <= 0:
                step = 1.0
            offset = float(repaired[loc[-1]]) + step

        if np.isfinite(repaired).all():
            out = out.copy()
            out["time_s"] = repaired
            warnings.append(
                "time_s rebuilt as monotonic global time from raw__mat_subrecord_index + local elapsed time (GV1 time fix v3)"
            )
            return out

    # Case 2: no subrecord IDs. As a last resort, rebuild from row index only when the
    # caller explicitly enabled row-index inference.
    # We do not access ReadOptions here, so this is intentionally not applied by default.
    return out

def standardize_dataframe(
    df: pd.DataFrame,
    options: ReadOptions | None = None,
    *,
    source_path: str | Path | None = None,
    source_format: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Convert a raw battery table to the GV1 standard table.

    The function is conservative: it keeps raw columns as ``raw__*`` columns by
    default, and it returns warnings instead of hiding ambiguous cases.
    """
    options = options or ReadOptions()
    metadata = metadata or {}
    warnings: list[str] = []

    if df.empty:
        raise ValueError("Cannot standardize an empty DataFrame")

    raw = df.copy()
    raw.columns = [_clean_name(c) for c in raw.columns]

    out = pd.DataFrame(index=raw.index)
    mapped_from: dict[str, str] = {}
    # Prefer explicit map, then alias/fuzzy. Avoid overwriting an already mapped standard column.
    for col in raw.columns:
        std = infer_standard_column(col, options.field_map)
        if std is None:
            continue
        if std in out.columns:
            warnings.append(f"Column {col!r} maps to {std!r}, already mapped from {mapped_from[std]!r}; skipped")
            continue
        try:
            converted = _convert_column(raw[col], col, std, options)
            if std in {"time_s", "current_A", "voltage_V", "capacity_Ah", "energy_Wh", "temperature_C"}:
                if pd.to_numeric(converted, errors="coerce").notna().sum() == 0:
                    warnings.append(f"Column {col!r} maps to {std!r} but has no numeric values; skipped")
                    continue
            out[std] = converted
            mapped_from[std] = col
        except Exception as exc:
            warnings.append(f"Failed to convert column {col!r} as {std!r}: {exc}")

    # Fallback time if requested.
    if "time_s" not in out.columns or out["time_s"].isna().all():
        if options.infer_time_from_row_index and options.sample_frequency_hz:
            out["time_s"] = np.arange(len(raw), dtype=float) / float(options.sample_frequency_hz)
            warnings.append("time_s inferred from row index and sample_frequency_hz")
        else:
            warnings.append("time_s not found; set infer_time_from_row_index=True with sample_frequency_hz if needed")

    # Normalize output current sign.
    if "current_A" in out.columns:
        out["current_A"] = normalize_sign_current(out["current_A"], options.current_sign_convention)

    # Defaults from options / metadata / path.
    if options.dataset_id is not None:
        out["dataset_id"] = options.dataset_id
    elif "dataset_id" not in out.columns:
        out["dataset_id"] = metadata.get("dataset_id", pd.NA)

    if options.batch_id is not None:
        out["batch_id"] = options.batch_id
    elif "batch_id" not in out.columns:
        out["batch_id"] = metadata.get("batch_id", pd.NA)

    if options.battery_id is not None:
        out["battery_id"] = options.battery_id
    elif "battery_id" not in out.columns:
        out["battery_id"] = metadata.get("battery_id", pd.NA)

    if options.cycle_id is not None:
        out["cycle_id"] = int(options.cycle_id)
    elif "cycle_id" not in out.columns:
        out["cycle_id"] = metadata.get("cycle_id", pd.NA)

    if options.step_id is not None:
        out["step_id"] = int(options.step_id)
    elif "step_id" not in out.columns:
        out["step_id"] = pd.NA

    if options.step_type is not None:
        out["step_type"] = options.step_type
    elif "step_type" not in out.columns:
        if "current_A" in out.columns:
            out["step_type"] = _classify_step_type(out["current_A"])
        else:
            out["step_type"] = "unknown"

    if "temperature_C" not in out.columns or out["temperature_C"].isna().all():
        if options.default_temperature_C is not None:
            out["temperature_C"] = float(options.default_temperature_C)
            warnings.append(f"temperature_C missing; filled with default {options.default_temperature_C} C")

    if "source_file" not in out.columns:
        out["source_file"] = str(source_path) if source_path is not None else metadata.get("source_file", pd.NA)
    if "source_format" not in out.columns:
        out["source_format"] = source_format or metadata.get("source_format", pd.NA)

    # Keep raw columns with safe prefix for debugging.
    if options.preserve_raw_columns:
        for col in raw.columns:
            raw_col = f"raw__{col}"
            if raw_col not in out.columns and col not in out.columns:
                out[raw_col] = raw[col]

    # Repair XJTU/MATLAB-style concatenated records whose system_time jumps backwards.
    out = _repair_nonmonotonic_time_axis(out, warnings)

    # Stable column order: standard columns first, then raw/debug columns.
    first = [c for c in STANDARD_COLUMNS if c in out.columns]
    rest = [c for c in out.columns if c not in first]
    out = out[first + rest].reset_index(drop=True)

    valid, problems = validate_standard_table(out)
    if not valid:
        warnings.extend(problems)
        if options.strict:
            raise ValueError("Standard table validation failed: " + "; ".join(problems))
    return out, warnings


def validate_standard_table(df: pd.DataFrame) -> tuple[bool, list[str]]:
    problems: list[str] = []
    required = ["time_s", "current_A", "voltage_V", "temperature_C", "source_file"]
    for col in required:
        if col not in df.columns:
            problems.append(f"missing required column {col!r}")
    for col in ["time_s", "current_A", "voltage_V"]:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().sum() == 0:
            problems.append(f"column {col!r} has no numeric values")
    if "time_s" in df.columns:
        t = pd.to_numeric(df["time_s"], errors="coerce")
        if t.notna().sum() > 1:
            dt = t.dropna().diff().dropna()
            if (dt < 0).any():
                problems.append("time_s is not monotonic nondecreasing")
    return len(problems) == 0, problems
