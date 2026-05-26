"""Canonical table schema used by GV1 data discovery and readers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


STANDARD_COLUMNS = [
    "dataset_id",
    "batch_id",
    "battery_id",
    "cell_id",
    "cycle_id",
    "step_id",
    "step_type",
    "time_s",
    "current_A",
    "voltage_V",
    "temperature_C",
    "capacity_Ah",
    "source_file",
    "source_format",
]

REQUIRED_TIME_SERIES_COLUMNS = ["time_s", "current_A", "voltage_V"]

INDEX_COLUMNS = [
    "dataset_id",
    "dataset_root",
    "batch_id",
    "battery_id",
    "cell_id",
    "protocol_id",
    "protocol_hint",
    "observed_control_mode",
    "source_file",
    "relative_path",
    "source_format",
    "cycle_id_hint",
    "file_size_bytes",
    "mtime_iso",
    "is_selected",
    "notes",
]


@dataclass(frozen=True)
class SchemaValidationResult:
    ok: bool
    missing_required: list[str]
    missing_optional: list[str]
    extra_columns: list[str]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "missing_required": self.missing_required,
            "missing_optional": self.missing_optional,
            "extra_columns": self.extra_columns,
        }


def validate_standard_columns(columns: Iterable[str]) -> SchemaValidationResult:
    observed = set(str(c) for c in columns)
    required = set(REQUIRED_TIME_SERIES_COLUMNS)
    standard = set(STANDARD_COLUMNS)
    missing_required = sorted(required - observed)
    missing_optional = sorted((standard - required) - observed)
    extra_columns = sorted(observed - standard)
    return SchemaValidationResult(
        ok=not missing_required,
        missing_required=missing_required,
        missing_optional=missing_optional,
        extra_columns=extra_columns,
    )


def with_standard_metadata(row: Mapping, *, defaults: Mapping | None = None) -> dict:
    """Return a dict with all standard columns present."""
    defaults = defaults or {}
    out = {col: defaults.get(col) for col in STANDARD_COLUMNS}
    out.update(dict(row))
    return out


def index_row_defaults() -> dict:
    return {col: None for col in INDEX_COLUMNS}
