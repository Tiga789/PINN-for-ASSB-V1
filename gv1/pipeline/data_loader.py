from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


def read_dataset_index(index_csv: str | Path) -> pd.DataFrame:
    p = Path(index_csv)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def read_standard_tables_from_index(
    index_csv: str | Path,
    *,
    adapter: str = 'xjtu',
    dataset_root: str | Path | None = None,
    max_files: int | None = None,
    default_temperature_C: float = 25.0,
) -> pd.DataFrame:
    """Read source files listed in a dataset index into one standard table."""
    idx = read_dataset_index(index_csv)
    if 'is_selected' in idx:
        selected = idx[idx['is_selected'].astype(str).str.lower().isin(['true', '1', 'yes'])].copy()
    else:
        selected = idx.copy()
    if max_files is not None:
        selected = selected.head(int(max_files))
    frames = []
    if adapter.lower() == 'xjtu':
        from gv1.adapters.xjtu_adapter import read_xjtu_file
        for _, row in selected.iterrows():
            res = read_xjtu_file(
                row['source_file'],
                dataset_root=dataset_root or row.get('dataset_root'),
                dataset_id=row.get('dataset_id', 'XJTU'),
                batch_id=row.get('batch_id'),
                battery_id=row.get('battery_id'),
                default_temperature_C=default_temperature_C,
            )
            frames.append(res.dataframe)
    else:
        from gv1.io.auto_reader import read_battery_file
        from gv1.io.reader_base import ReadOptions
        for _, row in selected.iterrows():
            res = read_battery_file(
                row['source_file'],
                ReadOptions(
                    dataset_id=row.get('dataset_id'),
                    batch_id=row.get('batch_id'),
                    battery_id=row.get('battery_id'),
                    default_temperature_C=default_temperature_C,
                    infer_time_from_row_index=True,
                    sample_frequency_hz=1.0,
                ),
            )
            frames.append(res.dataframe)
    if not frames:
        raise ValueError('No files were read from dataset index')
    return pd.concat(frames, ignore_index=True, sort=False)
