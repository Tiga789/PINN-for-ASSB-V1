from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd


def to_numeric_series(values: Any) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce")


def normalize_time_to_seconds(values: Any, unit: Optional[str] = None, raw_name: str = "") -> pd.Series:
    raw_series = pd.Series(values)
    s = to_numeric_series(values)
    # If the column is timestamp-like (e.g. system_time), convert to elapsed seconds.
    if s.notna().sum() == 0:
        dt = pd.to_datetime(raw_series.astype("string").str.replace(",", " ", regex=False), errors="coerce")
        if dt.notna().sum() > 0:
            first = dt[dt.notna()].iloc[0]
            return (dt - first).dt.total_seconds()
    u = (unit or "").strip().lower()
    name = raw_name.lower()
    if not u:
        if "min" in name or "minute" in name:
            u = "min"
        elif "ms" in name or "millisecond" in name:
            u = "ms"
        elif "hour" in name or name.endswith("_h"):
            u = "h"
        else:
            u = "s"
    if u in {"s", "sec", "second", "seconds"}:
        return s
    if u in {"min", "minute", "minutes"}:
        return s * 60.0
    if u in {"ms", "millisecond", "milliseconds"}:
        return s / 1000.0
    if u in {"h", "hr", "hour", "hours"}:
        return s * 3600.0
    raise ValueError(f"Unsupported time unit: {unit!r} for column {raw_name!r}")


def normalize_current_to_ampere(values: Any, unit: Optional[str] = None, raw_name: str = "") -> pd.Series:
    s = to_numeric_series(values)
    u = (unit or "").strip().lower()
    name = raw_name.lower()
    if not u:
        if "ma" in name and "mah" not in name:
            u = "ma"
        else:
            u = "a"
    if u in {"a", "amp", "ampere", "amperes"}:
        return s
    if u in {"ma", "milliamp", "milliampere", "milliamperes"}:
        return s / 1000.0
    raise ValueError(f"Unsupported current unit: {unit!r} for column {raw_name!r}")


def normalize_voltage_to_volt(values: Any, unit: Optional[str] = None, raw_name: str = "") -> pd.Series:
    s = to_numeric_series(values)
    u = (unit or "").strip().lower()
    name = raw_name.lower()
    if not u:
        if "mv" in name:
            u = "mv"
        else:
            u = "v"
    if u in {"v", "volt", "volts"}:
        return s
    if u in {"mv", "millivolt", "millivolts"}:
        return s / 1000.0
    raise ValueError(f"Unsupported voltage unit: {unit!r} for column {raw_name!r}")


def normalize_capacity_to_ah(values: Any, unit: Optional[str] = None, raw_name: str = "") -> pd.Series:
    s = to_numeric_series(values)
    u = (unit or "").strip().lower()
    name = raw_name.lower()
    if not u:
        if "mah" in name:
            u = "mah"
        else:
            u = "ah"
    if u in {"ah", "a*h", "ampere-hour", "ampere-hours"}:
        return s
    if u in {"mah", "milliampere-hour", "milliampere-hours"}:
        return s / 1000.0
    raise ValueError(f"Unsupported capacity unit: {unit!r} for column {raw_name!r}")


def normalize_energy_to_wh(values: Any, unit: Optional[str] = None, raw_name: str = "") -> pd.Series:
    s = to_numeric_series(values)
    u = (unit or "").strip().lower()
    name = raw_name.lower()
    if not u:
        if "mwh" in name:
            u = "mwh"
        else:
            u = "wh"
    if u in {"wh", "w*h", "watt-hour", "watt-hours"}:
        return s
    if u in {"mwh", "milliwatt-hour", "milliwatt-hours"}:
        return s / 1000.0
    raise ValueError(f"Unsupported energy unit: {unit!r} for column {raw_name!r}")


def normalize_temperature_to_celsius(values: Any, unit: Optional[str] = None, raw_name: str = "") -> pd.Series:
    s = to_numeric_series(values)
    u = (unit or "").strip().lower()
    name = raw_name.lower()
    if not u:
        if "_k" in name or name.endswith("k"):
            u = "k"
        else:
            u = "c"
    if u in {"c", "degc", "celsius", "°c"}:
        return s
    if u in {"k", "kelvin"}:
        return s - 273.15
    raise ValueError(f"Unsupported temperature unit: {unit!r} for column {raw_name!r}")


def normalize_sign_current(series: pd.Series, convention: str) -> pd.Series:
    c = convention.lower().strip()
    if c == "positive_is_charge" or c == "as_recorded":
        return series
    if c == "positive_is_discharge":
        return -series
    raise ValueError(
        "current_sign_convention must be 'positive_is_charge', 'positive_is_discharge', or 'as_recorded'"
    )


def safe_diff(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if len(arr) == 0:
        return pd.Series([], dtype=float)
    out = np.empty_like(arr)
    out[0] = math.nan
    out[1:] = np.diff(arr)
    return pd.Series(out, index=values.index)
