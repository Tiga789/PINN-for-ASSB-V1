"""XJTU dataset adapter for GV1 measured-current replay.

This adapter is deliberately thin: it adds XJTU-specific metadata and protocol
labels around the generic GV1 data readers.  It does not bind the code to a
specific battery file; target batches/cells are selected by manifests and the
dataset-index layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:
    from gv1.io.auto_reader import read_battery_file
    from gv1.io.reader_base import ReadOptions, ReadResult
except Exception:  # pragma: no cover - allows static inspection before GV1 IO is installed
    read_battery_file = None  # type: ignore
    ReadOptions = None  # type: ignore
    ReadResult = Any  # type: ignore

try:
    from gv1.cell_id_parser import parse_cell_id_info
except Exception:  # pragma: no cover
    parse_cell_id_info = None  # type: ignore

from .xjtu_protocols import get_xjtu_protocol


@dataclass(frozen=True)
class XJTUFileMetadata:
    dataset_id: str
    batch_id: str | None
    battery_id: str | None
    cell_id: str
    protocol_id: str
    observed_control_mode: str
    source_file: str
    relative_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _search_batch_from_path(path: Path) -> str | None:
    for part in path.parts:
        m = re.search(r"Batch[-_ ]?(\d+)", part, flags=re.IGNORECASE)
        if m:
            return f"Batch-{int(m.group(1))}"
    return None


def _search_battery_from_name(name: str) -> str | None:
    # Examples: 2C_battery-1.mat, R2.5_battery-1_cycle_0001.csv
    m = re.search(r"battery[-_ ]?(\d+)", name, flags=re.IGNORECASE)
    if m:
        return f"battery-{int(m.group(1))}"
    m = re.search(r"cell[-_ ]?(\d+)", name, flags=re.IGNORECASE)
    if m:
        return f"cell-{int(m.group(1))}"
    return None


def _relative(path: Path, dataset_root: str | Path | None) -> str | None:
    if dataset_root is None:
        return None
    try:
        return str(path.resolve().relative_to(Path(dataset_root).resolve())).replace("\\", "/")
    except Exception:
        return None


def parse_xjtu_file_metadata(
    path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    dataset_id: str = "XJTU",
    batch_id: str | None = None,
    battery_id: str | None = None,
) -> XJTUFileMetadata:
    p = Path(path)
    inferred_batch = batch_id or _search_batch_from_path(p)
    inferred_battery = battery_id or _search_battery_from_name(p.name)

    if parse_cell_id_info is not None:
        try:
            parsed = parse_cell_id_info(str(p), dataset_id=dataset_id, dataset_root=dataset_root)
            inferred_batch = inferred_batch or parsed.batch_id
            inferred_battery = inferred_battery or parsed.battery_id
            cell_id = parsed.cell_id
        except Exception:
            cell_id = "_".join(x for x in [dataset_id, inferred_batch, inferred_battery] if x)
    else:
        cell_id = "_".join(x for x in [dataset_id, inferred_batch, inferred_battery] if x)

    protocol = get_xjtu_protocol(inferred_batch)
    return XJTUFileMetadata(
        dataset_id=dataset_id,
        batch_id=inferred_batch,
        battery_id=inferred_battery,
        cell_id=cell_id or f"{dataset_id}_unknown_cell",
        protocol_id=protocol.protocol_id,
        observed_control_mode=protocol.observed_control_mode,
        source_file=str(p),
        relative_path=_relative(p, dataset_root),
    )


def _default_field_map() -> dict[str, str]:
    return {
        "relative_time_min": "time_s",
        "voltage_V": "voltage_V",
        "current_A": "current_A",
        "capacity_Ah": "capacity_Ah",
        "power_Wh": "energy_Wh",
        "temperature_C": "temperature_C",
        "description": "step_type",
        "cycle": "cycle_id",
        "cycle_id": "cycle_id",
        "step": "step_id",
        "step_id": "step_id",
    }


def _default_unit_map() -> dict[str, str]:
    # The sample CSV exported from the XJTU .mat data uses relative_time_min.
    return {
        "relative_time_min": "min",
        "time_s": "s",
        "current_A": "A",
        "voltage_V": "V",
        "capacity_Ah": "Ah",
        "power_Wh": "Wh",
        "energy_Wh": "Wh",
        "temperature_C": "C",
    }


def make_xjtu_read_options(
    metadata: XJTUFileMetadata,
    *,
    default_temperature_C: float = 25.0,
    field_map: Mapping[str, str] | None = None,
    unit_map: Mapping[str, str] | None = None,
    mat_table_path: str | None = None,
) -> Any:
    if ReadOptions is None:
        raise ImportError("gv1.io.reader_base.ReadOptions is required. Install/unzip GV1_通用数据读取层_v1 first.")
    fm = _default_field_map()
    if field_map:
        fm.update(dict(field_map))
    um = _default_unit_map()
    if unit_map:
        um.update(dict(unit_map))
    return ReadOptions(
        dataset_id=metadata.dataset_id,
        batch_id=metadata.batch_id,
        battery_id=metadata.battery_id,
        current_sign_convention="positive_is_charge",
        default_temperature_C=default_temperature_C,
        sample_frequency_hz=1.0,
        infer_time_from_row_index=True,
        field_map=fm,
        unit_map=um,
        mat_table_path=mat_table_path,
        preserve_raw_columns=True,
    )


def _refine_step_type(df: pd.DataFrame, *, current_threshold_A: float = 1e-9) -> pd.DataFrame:
    out = df.copy()
    cur = pd.to_numeric(out.get("current_A"), errors="coerce") if "current_A" in out else pd.Series(np.nan, index=out.index)
    volt = pd.to_numeric(out.get("voltage_V"), errors="coerce") if "voltage_V" in out else pd.Series(np.nan, index=out.index)
    step = pd.Series("unknown", index=out.index, dtype="string")
    step[cur > current_threshold_A] = "charge"
    step[cur < -current_threshold_A] = "discharge"
    step[cur.abs() <= current_threshold_A] = "rest"
    # Observed CV charge region: positive current, near 4.2V, current often decays.
    cv_mask = (cur > current_threshold_A) & (volt >= 4.18)
    step[cv_mask] = "charge_cv_observed"
    out["step_type_auto"] = step
    if "step_type" not in out or out["step_type"].isna().all():
        out["step_type"] = step
    return out


def attach_xjtu_metadata(df: pd.DataFrame, metadata: XJTUFileMetadata) -> pd.DataFrame:
    out = df.copy()
    out["dataset_id"] = metadata.dataset_id
    if metadata.batch_id is not None:
        out["batch_id"] = metadata.batch_id
    if metadata.battery_id is not None:
        out["battery_id"] = metadata.battery_id
    out["cell_id"] = metadata.cell_id
    out["protocol_id"] = metadata.protocol_id
    out["observed_control_mode"] = metadata.observed_control_mode
    out["current_input_mode"] = "measured_current_profile"
    return out


def standardize_xjtu_dataframe(df: pd.DataFrame, metadata: XJTUFileMetadata) -> pd.DataFrame:
    out = attach_xjtu_metadata(df, metadata)
    out = _refine_step_type(out)
    # Make cycle_id robust for single-cycle exports. Full .mat files should contain cycle information if present.
    if "cycle_id" not in out or out["cycle_id"].isna().all():
        # Try filename suffix cycle_0001.
        m = re.search(r"cycle[_-]?(\d+)", Path(metadata.source_file).name, flags=re.IGNORECASE)
        if m:
            out["cycle_id"] = int(m.group(1))
    return out


def read_xjtu_file(
    path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    dataset_id: str = "XJTU",
    batch_id: str | None = None,
    battery_id: str | None = None,
    default_temperature_C: float = 25.0,
    mat_table_path: str | None = None,
    field_map: Mapping[str, str] | None = None,
    unit_map: Mapping[str, str] | None = None,
) -> Any:
    """Read one XJTU file into the GV1 standard table plus XJTU metadata."""
    if read_battery_file is None:
        raise ImportError("gv1.io.auto_reader.read_battery_file is required. Install/unzip GV1_通用数据读取层_v1 first.")
    meta = parse_xjtu_file_metadata(path, dataset_root=dataset_root, dataset_id=dataset_id, batch_id=batch_id, battery_id=battery_id)
    options = make_xjtu_read_options(
        meta,
        default_temperature_C=default_temperature_C,
        mat_table_path=mat_table_path,
        field_map=field_map,
        unit_map=unit_map,
    )
    result = read_battery_file(path, options)
    df = standardize_xjtu_dataframe(result.dataframe, meta)
    result.dataframe = df
    result.metadata.update({"xjtu": meta.to_dict()})
    return result


class XJTUAdapter:
    """Small object wrapper for repeated XJTU reads from one dataset root."""

    def __init__(self, dataset_root: str | Path, *, dataset_id: str = "XJTU", default_temperature_C: float = 25.0):
        self.dataset_root = Path(dataset_root)
        self.dataset_id = dataset_id
        self.default_temperature_C = float(default_temperature_C)

    def metadata(self, path: str | Path, *, batch_id: str | None = None, battery_id: str | None = None) -> XJTUFileMetadata:
        return parse_xjtu_file_metadata(path, dataset_root=self.dataset_root, dataset_id=self.dataset_id, batch_id=batch_id, battery_id=battery_id)

    def read(self, path: str | Path, **kwargs: Any) -> Any:
        return read_xjtu_file(
            path,
            dataset_root=self.dataset_root,
            dataset_id=self.dataset_id,
            default_temperature_C=self.default_temperature_C,
            **kwargs,
        )
