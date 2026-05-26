from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .reader_base import ReadResult


def write_standard_cache(
    result: ReadResult,
    output_path: str | Path,
    *,
    format: Literal["parquet", "csv"] | None = None,
    write_metadata_json: bool = True,
) -> Path:
    """Write a standardized GV1 table to Parquet or CSV.

    Parquet is recommended for full-batch training; CSV is useful for manual inspection.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fmt = format or ("parquet" if p.suffix.lower() in {".parquet", ".pq"} else "csv")
    if fmt == "parquet":
        try:
            result.dataframe.to_parquet(p, index=False)
        except ImportError as exc:
            raise ImportError("Writing parquet requires pyarrow or fastparquet. Install pyarrow.") from exc
    elif fmt == "csv":
        result.dataframe.to_csv(p, index=False, encoding="utf-8-sig")
    else:
        raise ValueError("format must be 'parquet' or 'csv'")
    if write_metadata_json:
        meta_path = p.with_suffix(p.suffix + ".metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_path": result.source_path,
                    "source_format": result.source_format,
                    "n_rows": int(len(result.dataframe)),
                    "columns": list(result.dataframe.columns),
                    "warnings": result.warnings,
                    "metadata": result.metadata,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    return p
