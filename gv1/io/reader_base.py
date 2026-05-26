from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd


STANDARD_COLUMNS = [
    "dataset_id",
    "batch_id",
    "battery_id",
    "cycle_id",
    "step_id",
    "step_type",
    "time_s",
    "current_A",
    "voltage_V",
    "temperature_C",
    "capacity_Ah",
    "energy_Wh",
    "source_file",
    "source_format",
]


@dataclass
class ReadOptions:
    """Options used by all GV1 readers.

    The output convention is always ``positive current = charge`` unless
    ``current_sign_convention`` is set to ``"as_recorded"``.
    """

    dataset_id: Optional[str] = None
    batch_id: Optional[str] = None
    battery_id: Optional[str] = None
    cycle_id: Optional[int] = None
    step_id: Optional[int] = None
    step_type: Optional[str] = None

    # Current sign in the raw file. Output is normalized to positive charge.
    current_sign_convention: str = "positive_is_charge"  # positive_is_charge | positive_is_discharge | as_recorded

    # Temperature handling.
    default_temperature_C: Optional[float] = 25.0

    # Time fallback. XJTU is 1 Hz, but for generic data this should be set in a manifest.
    sample_frequency_hz: Optional[float] = None
    infer_time_from_row_index: bool = False

    # Explicit mapping from raw field names to standard field names.
    # Example: {"Relative_Time(min)": "time_s", "Voltage": "voltage_V"}
    field_map: Mapping[str, str] = field(default_factory=dict)

    # Raw unit hints keyed by raw column name or standard column name.
    # Example: {"time": "min", "current": "mA"}
    unit_map: Mapping[str, str] = field(default_factory=dict)

    preserve_raw_columns: bool = True
    strict: bool = False
    mat_table_path: Optional[str] = None


@dataclass
class ReadResult:
    dataframe: pd.DataFrame
    source_path: str
    source_format: str
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def require_valid(self) -> "ReadResult":
        from .field_mapper import validate_standard_table

        valid, problems = validate_standard_table(self.dataframe)
        if not valid:
            msg = "; ".join(problems)
            raise ValueError(f"Standard table validation failed for {self.source_path}: {msg}")
        return self


def ensure_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Input file does not exist: {p}")
    if not p.is_file():
        raise IsADirectoryError(f"Expected a file, got directory: {p}")
    return p
