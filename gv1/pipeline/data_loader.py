from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd

# Columns needed by measured-current replay.  Reading only these columns avoids
# loading the raw debug columns from large XJTU Parquet caches.
REPLAY_PROFILE_COLUMNS: list[str] = [
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

_NUMERIC_FLOAT32 = [
    "current_A",
    "voltage_V",
    "temperature_C",
    "capacity_Ah",
    "energy_Wh",
]
_NUMERIC_FLOAT64 = ["time_s"]
_INT_COLUMNS = ["cycle_id", "step_id"]
_STRING_COLUMNS = [
    "dataset_id",
    "batch_id",
    "battery_id",
    "cell_id",
    "cell_uid",
    "protocol",
    "protocol_id",
    "observed_control_mode",
    "current_input_mode",
    "step_type",
    "source_file",
    "source_format",
]


def read_dataset_index(index_csv: str | Path) -> pd.DataFrame:
    p = Path(index_csv)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def _row_get(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    try:
        value = row.get(key, default)  # type: ignore[attr-defined]
    except AttributeError:
        value = default
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return default
    return value


def _selected_rows(index_csv: str | Path, max_files: int | None = None) -> pd.DataFrame:
    idx = read_dataset_index(index_csv)
    if "is_selected" in idx.columns:
        selected = idx[idx["is_selected"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    else:
        selected = idx.copy()
    if max_files is not None:
        selected = selected.head(int(max_files))
    return selected.reset_index(drop=True)


def _available_parquet_columns(path: str | Path) -> list[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        schema = pq.ParquetFile(path).schema_arrow
        return list(schema.names)
    except Exception:
        # Fallback: let pandas read metadata / small schema when possible.
        try:
            return list(pd.read_parquet(path, columns=[]).columns)
        except Exception:
            return []


def _read_standard_parquet_light(path: str | Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    p = Path(path)
    requested = list(columns or REPLAY_PROFILE_COLUMNS)
    available = _available_parquet_columns(p)
    if available:
        cols = [c for c in requested if c in available]
        # If source_file/source_format were not requested but exist, keep them for provenance.
        for extra in ["source_file", "source_format"]:
            if extra in available and extra not in cols:
                cols.append(extra)
        if not cols:
            raise ValueError(f"No requested columns are present in parquet file: {p}")
        df = pd.read_parquet(p, columns=cols)
    else:
        df = pd.read_parquet(p)
        keep = [c for c in requested if c in df.columns]
        if keep:
            df = df[keep]
    return _optimize_standard_dataframe(df)


def _optimize_standard_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in _NUMERIC_FLOAT64:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    for c in _NUMERIC_FLOAT32:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    for c in _INT_COLUMNS:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors="coerce")
            if s.notna().all():
                out[c] = s.astype("int32")
            else:
                out[c] = s.astype("Int32")
    for c in _STRING_COLUMNS:
        if c in out.columns:
            out[c] = out[c].astype("string")
    return out


def _ensure_metadata_columns(df: pd.DataFrame, row: Mapping[str, Any], source_path: str | Path) -> pd.DataFrame:
    out = df.copy()
    for col in ["dataset_id", "batch_id", "battery_id", "protocol"]:
        val = _row_get(row, col, None)
        if val is not None and (col not in out.columns or out[col].isna().all()):
            out[col] = str(val)
    if "source_file" not in out.columns or out["source_file"].isna().all():
        out["source_file"] = str(source_path)
    if "source_format" not in out.columns or out["source_format"].isna().all():
        out["source_format"] = Path(source_path).suffix.lower().lstrip(".") or "unknown"
    if "current_input_mode" not in out.columns:
        out["current_input_mode"] = "measured_current_profile"
    return out


def read_one_standard_table(
    row: Mapping[str, Any],
    *,
    adapter: str = "xjtu",
    dataset_root: str | Path | None = None,
    default_temperature_C: float = 25.0,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read exactly one indexed source file into a standard table.

    For .parquet caches this uses a light column subset to avoid ArrowMemoryError
    on multi-million-row XJTU files. Raw .mat/.csv still go through the existing
    GV1 readers, but only one file is held in memory at a time.
    """
    source_file = _row_get(row, "source_file") or _row_get(row, "file_path") or _row_get(row, "path")
    if not source_file:
        raise ValueError(f"Index row does not contain source_file/file_path/path: {row}")
    p = Path(str(source_file))
    suffix = p.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = _read_standard_parquet_light(p, columns=columns)
        return _ensure_metadata_columns(df, row, p)

    if adapter.lower() == "xjtu":
        from gv1.adapters.xjtu_adapter import read_xjtu_file

        res = read_xjtu_file(
            p,
            dataset_root=dataset_root or _row_get(row, "dataset_root"),
            dataset_id=_row_get(row, "dataset_id", "XJTU"),
            batch_id=_row_get(row, "batch_id"),
            battery_id=_row_get(row, "battery_id"),
            default_temperature_C=default_temperature_C,
        )
        return _optimize_standard_dataframe(res.dataframe)

    from gv1.io.auto_reader import read_battery_file
    from gv1.io.reader_base import ReadOptions

    res = read_battery_file(
        p,
        ReadOptions(
            dataset_id=_row_get(row, "dataset_id"),
            batch_id=_row_get(row, "batch_id"),
            battery_id=_row_get(row, "battery_id"),
            default_temperature_C=default_temperature_C,
            infer_time_from_row_index=True,
            sample_frequency_hz=1.0,
            preserve_raw_columns=False,
        ),
        fallback=False,
    )
    return _optimize_standard_dataframe(res.dataframe)


def iter_standard_tables_from_index(
    index_csv: str | Path,
    *,
    adapter: str = "xjtu",
    dataset_root: str | Path | None = None,
    max_files: int | None = None,
    default_temperature_C: float = 25.0,
    columns: Sequence[str] | None = None,
) -> Iterator[tuple[dict[str, Any], pd.DataFrame]]:
    """Yield one standard table at a time from an index CSV."""
    selected = _selected_rows(index_csv, max_files=max_files)
    for _, row in selected.iterrows():
        row_dict = row.to_dict()
        df = read_one_standard_table(
            row_dict,
            adapter=adapter,
            dataset_root=dataset_root,
            default_temperature_C=default_temperature_C,
            columns=columns,
        )
        yield row_dict, df


def read_standard_tables_from_index(
    index_csv: str | Path,
    *,
    adapter: str = "xjtu",
    dataset_root: str | Path | None = None,
    max_files: int | None = None,
    default_temperature_C: float = 25.0,
) -> pd.DataFrame:
    """Read source files listed in a dataset index into one standard table.

    Kept for backward compatibility.  New large-data workflows should prefer
    iter_standard_tables_from_index so only one cell file is loaded at a time.
    """
    frames = [
        df
        for _, df in iter_standard_tables_from_index(
            index_csv,
            adapter=adapter,
            dataset_root=dataset_root,
            max_files=max_files,
            default_temperature_C=default_temperature_C,
        )
    ]
    if not frames:
        raise ValueError("No files were read from dataset index")
    return pd.concat(frames, ignore_index=True, sort=False)
