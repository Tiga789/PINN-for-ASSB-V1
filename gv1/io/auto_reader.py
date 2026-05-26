from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .csv_reader import read_csv_battery_file
from .mat_reader import read_mat_battery_file
from .parquet_reader import read_parquet_battery_file
from .reader_base import ReadOptions, ReadResult, ensure_path


def read_battery_file(path: str | Path, options: ReadOptions | None = None, *, fallback: bool = True) -> ReadResult:
    """Read a .mat/.csv/.parquet battery data file into a standard DataFrame."""
    p = ensure_path(path)
    options = options or ReadOptions()
    suffix = p.suffix.lower()
    readers = []
    if suffix == ".csv":
        readers = [read_csv_battery_file, read_parquet_battery_file, read_mat_battery_file]
    elif suffix in {".parquet", ".pq"}:
        readers = [read_parquet_battery_file, read_csv_battery_file, read_mat_battery_file]
    elif suffix == ".mat":
        readers = [read_mat_battery_file, read_csv_battery_file, read_parquet_battery_file]
    else:
        readers = [read_mat_battery_file, read_csv_battery_file, read_parquet_battery_file]
    if not fallback:
        readers = readers[:1]
    errors: list[str] = []
    for reader in readers:
        try:
            return reader(p, options)
        except Exception as exc:
            errors.append(f"{reader.__name__}: {type(exc).__name__}: {exc}")
    raise ValueError(f"All readers failed for {p}: {' | '.join(errors)}")


def read_battery_files(paths: Sequence[str | Path], options: ReadOptions | None = None) -> ReadResult:
    """Read many files and concatenate their standardized outputs."""
    if not paths:
        raise ValueError("No input paths provided")
    frames = []
    warnings: list[str] = []
    metadata = {"files": []}
    for path in paths:
        result = read_battery_file(path, options)
        frames.append(result.dataframe)
        warnings.extend(result.warnings)
        metadata["files"].append(
            {
                "source_path": result.source_path,
                "source_format": result.source_format,
                "n_rows": int(len(result.dataframe)),
                "metadata": result.metadata,
            }
        )
    df = pd.concat(frames, ignore_index=True, sort=False)
    return ReadResult(df, "<multiple>", "mixed", metadata, warnings)
