"""Build and persist a generic GV1 dataset index."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .cell_id_parser import parse_cell_id_info
from .data_discovery import DiscoveredFile, discover_files, summarize_discovery
from .protocol_parser import protocol_from_hint
from .standard_table import INDEX_COLUMNS, index_row_defaults


@dataclass(frozen=True)
class DatasetIndexRow:
    dataset_id: str
    dataset_root: str
    batch_id: str | None
    battery_id: str | None
    cell_id: str
    protocol_id: str
    protocol_hint: str | None
    observed_control_mode: str
    source_file: str
    relative_path: str
    source_format: str
    cycle_id_hint: int | None
    file_size_bytes: int
    mtime_iso: str
    is_selected: bool = True
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def build_index_rows(
    discovered_files: Sequence[DiscoveredFile],
    *,
    dataset_id: str,
    protocol_mapping: Mapping | None = None,
) -> list[DatasetIndexRow]:
    rows: list[DatasetIndexRow] = []
    for f in discovered_files:
        ids = parse_cell_id_info(f.source_file, dataset_id=dataset_id, dataset_root=f.dataset_root)
        proto = protocol_from_hint(
            ids.protocol_hint,
            batch_id=ids.batch_id,
            protocol_mapping=protocol_mapping,
        )
        rows.append(
            DatasetIndexRow(
                dataset_id=dataset_id,
                dataset_root=f.dataset_root,
                batch_id=ids.batch_id,
                battery_id=ids.battery_id,
                cell_id=ids.cell_id,
                protocol_id=proto.protocol_id,
                protocol_hint=proto.protocol_hint,
                observed_control_mode=proto.observed_control_mode,
                source_file=f.source_file,
                relative_path=f.relative_path,
                source_format=f.source_format,
                cycle_id_hint=ids.cycle_id_hint,
                file_size_bytes=f.file_size_bytes,
                mtime_iso=f.mtime_iso,
                is_selected=True,
                notes=proto.notes,
            )
        )
    return rows


def rows_to_dicts(rows: Iterable[DatasetIndexRow]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        d = index_row_defaults()
        d.update(row.to_dict())
        out.append(d)
    return out


def write_index_csv(rows: Sequence[DatasetIndexRow], output_csv: str | Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for d in rows_to_dicts(rows):
            writer.writerow(d)


def write_index_jsonl(rows: Sequence[DatasetIndexRow], output_jsonl: str | Path) -> None:
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for d in rows_to_dicts(rows):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def read_index_csv(input_csv: str | Path) -> list[dict]:
    with Path(input_csv).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def summarize_index(rows: Sequence[DatasetIndexRow]) -> dict:
    def inc(counter: dict, key: str | None) -> None:
        k = key if key not in (None, "") else "unknown"
        counter[k] = counter.get(k, 0) + 1

    by_batch: dict[str, int] = {}
    by_battery: dict[str, int] = {}
    by_format: dict[str, int] = {}
    by_protocol: dict[str, int] = {}
    by_control_mode: dict[str, int] = {}
    for row in rows:
        inc(by_batch, row.batch_id)
        inc(by_battery, row.battery_id)
        inc(by_format, row.source_format)
        inc(by_protocol, row.protocol_id)
        inc(by_control_mode, row.observed_control_mode)
    return {
        "row_count": len(rows),
        "cell_count": len({r.cell_id for r in rows}),
        "by_batch": dict(sorted(by_batch.items())),
        "by_battery": dict(sorted(by_battery.items())),
        "by_format": dict(sorted(by_format.items())),
        "by_protocol": dict(sorted(by_protocol.items())),
        "by_control_mode": dict(sorted(by_control_mode.items())),
    }


def build_dataset_index(
    *,
    dataset_root: str | Path,
    dataset_id: str,
    include_batches: Sequence[str] | None = None,
    file_patterns: Sequence[str] = ("*.mat", "*.csv", "*.parquet"),
    recursive: bool = True,
    protocol_mapping: Mapping | None = None,
    max_files: int | None = None,
) -> tuple[list[DatasetIndexRow], dict]:
    files = discover_files(
        dataset_root,
        file_patterns=file_patterns,
        include_batches=include_batches,
        recursive=recursive,
        max_files=max_files,
    )
    rows = build_index_rows(files, dataset_id=dataset_id, protocol_mapping=protocol_mapping)
    summary = {
        "discovery": summarize_discovery(files),
        "index": summarize_index(rows),
    }
    return rows, summary


def write_summary_json(summary: Mapping, output_json: str | Path) -> None:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
