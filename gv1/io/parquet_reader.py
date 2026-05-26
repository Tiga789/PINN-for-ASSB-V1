from __future__ import annotations

from pathlib import Path

import pandas as pd

from .field_mapper import standardize_dataframe
from .reader_base import ReadOptions, ReadResult, ensure_path


def read_parquet_battery_file(path: str | Path, options: ReadOptions | None = None) -> ReadResult:
    p = ensure_path(path)
    options = options or ReadOptions()
    try:
        df = pd.read_parquet(p)
    except ImportError as exc:
        raise ImportError("Reading parquet requires pyarrow or fastparquet. Install pyarrow for GV1 cache support.") from exc
    std, warnings = standardize_dataframe(df, options, source_path=p, source_format="parquet")
    metadata = {"reader": "parquet", "raw_shape": list(df.shape), "raw_columns": list(map(str, df.columns))}
    return ReadResult(std, str(p), "parquet", metadata, warnings)
