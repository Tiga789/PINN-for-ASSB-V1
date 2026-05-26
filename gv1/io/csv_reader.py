from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .field_mapper import standardize_dataframe
from .reader_base import ReadOptions, ReadResult, ensure_path


def read_csv_battery_file(path: str | Path, options: ReadOptions | None = None) -> ReadResult:
    p = ensure_path(path)
    options = options or ReadOptions()
    errors: list[str] = []
    df = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin1"):
        try:
            df = pd.read_csv(p, encoding=enc)
            break
        except Exception as exc:
            errors.append(f"encoding={enc}: {exc}")
    if df is None:
        raise ValueError(f"Failed to read CSV file {p}. Attempts: {' | '.join(errors)}")
    std, warnings = standardize_dataframe(df, options, source_path=p, source_format="csv")
    metadata = {
        "reader": "csv",
        "raw_shape": list(df.shape),
        "raw_columns": list(map(str, df.columns)),
        "csv_encoding_attempt_errors": errors[:-1] if len(errors) > 1 else [],
    }
    return ReadResult(std, str(p), "csv", metadata, warnings)
