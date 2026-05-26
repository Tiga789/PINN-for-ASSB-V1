from __future__ import annotations

from pathlib import Path

import pandas as pd

from .field_mapper import standardize_dataframe
from .reader_base import ReadOptions, ReadResult, ensure_path

DEFAULT_PARQUET_COLUMNS = [
    "dataset_id",
    "batch_id",
    "battery_id",
    "cell_id",
    "cell_uid",
    "protocol",
    "protocol_id",
    "observed_control_mode",
    "current_input_mode",
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


def _schema_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return list(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return []


def read_parquet_battery_file(path: str | Path, options: ReadOptions | None = None) -> ReadResult:
    p = ensure_path(path)
    options = options or ReadOptions()
    try:
        available = _schema_columns(p)
        if available:
            cols = [c for c in DEFAULT_PARQUET_COLUMNS if c in available]
            df = pd.read_parquet(p, columns=cols or None)
        else:
            df = pd.read_parquet(p)
    except ImportError as exc:
        raise ImportError("Reading parquet requires pyarrow or fastparquet. Install pyarrow for GV1 cache support.") from exc

    std, warnings = standardize_dataframe(df, options, source_path=p, source_format="parquet")
    metadata = {"reader": "parquet", "raw_shape": list(df.shape), "raw_columns": list(map(str, df.columns))}
    return ReadResult(std, str(p), "parquet", metadata, warnings)
