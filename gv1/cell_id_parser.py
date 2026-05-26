"""Generic cell/batch/cycle identifier parsing for GV1 datasets.

This module intentionally avoids dataset-specific assumptions.  It extracts
common identifiers from file paths such as:

    E:/XJTU battery dataset/Batch-1/2C_battery-1.mat
    .../Batch-3/R2.5_battery-1_cycle_0001.csv

Dataset-specific semantics should live in an adapter layer, not here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Optional


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_identifier(value: object, *, default: str = "unknown") -> str:
    """Return a filesystem/model-safe identifier string."""
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    text = text.replace("\\", "/")
    text = _SAFE_ID_RE.sub("_", text)
    text = re.sub(r"_+", "_", text).strip("_.-")
    return text or default


def _canonical_numbered(prefix: str, number: str) -> str:
    try:
        n = int(number)
        return f"{prefix}-{n}"
    except Exception:
        return f"{prefix}-{number}"


def parse_batch_id(path: str | Path) -> Optional[str]:
    """Parse a batch identifier from any path component.

    Accepted examples: ``Batch-1``, ``batch_01``, ``BATCH 3``.
    """
    p = Path(path)
    for part in list(p.parts)[::-1]:
        m = re.search(r"\bbatch\s*[-_ ]?\s*(\d+)\b", part, flags=re.IGNORECASE)
        if m:
            return _canonical_numbered("Batch", m.group(1))
    return None


def parse_battery_id(path: str | Path) -> Optional[str]:
    """Parse a battery/cell identifier from a path or filename.

    Accepted examples: ``battery-1``, ``battery_001``, ``cell-4``.
    """
    text = "/".join(Path(path).parts)
    patterns = [
        r"(?:^|[^A-Za-z0-9])battery\s*[-_ ]?\s*(\d+)(?:\b|[^A-Za-z0-9])",
        r"(?:^|[^A-Za-z0-9])cell\s*[-_ ]?\s*(\d+)(?:\b|[^A-Za-z0-9])",
        r"(?:^|[^A-Za-z0-9])cellid\s*[-_ ]?\s*(\d+)(?:\b|[^A-Za-z0-9])",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            label = "battery" if "battery" in pat else "cell"
            return _canonical_numbered(label, m.group(1))
    return None


def parse_cycle_id_hint(path: str | Path) -> Optional[int]:
    """Parse a cycle id from filename if present."""
    name = Path(path).name
    patterns = [r"(?:^|[^A-Za-z0-9])cycle\s*[-_ ]?\s*(\d+)(?:\b|[^A-Za-z0-9])", r"(?:^|[^A-Za-z0-9])cyc\s*[-_ ]?\s*(\d+)(?:\b|[^A-Za-z0-9])"]
    for pat in patterns:
        m = re.search(pat, name, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def infer_protocol_prefix_from_name(path: str | Path) -> Optional[str]:
    """Return a likely protocol prefix before ``battery``/``cell`` tokens.

    Examples:
      ``2C_battery-1.mat`` -> ``2C``
      ``R2.5_battery-1_cycle_0001.csv`` -> ``R2.5``
    """
    stem = Path(path).stem
    m = re.match(r"(.+?)[-_ ]+(?:battery|cell)\s*[-_ ]?\s*\d+", stem, flags=re.IGNORECASE)
    if m:
        return sanitize_identifier(m.group(1), default="") or None
    return None


@dataclass(frozen=True)
class CellIdInfo:
    """Structured identifiers inferred from a source file path."""

    source_file: str
    dataset_id: Optional[str]
    batch_id: Optional[str]
    battery_id: Optional[str]
    cell_id: str
    protocol_hint: Optional[str]
    cycle_id_hint: Optional[int]

    def to_dict(self) -> dict:
        return asdict(self)


def parse_cell_id_info(
    path: str | Path,
    *,
    dataset_id: Optional[str] = None,
    dataset_root: Optional[str | Path] = None,
) -> CellIdInfo:
    """Infer dataset/batch/battery identifiers from a path.

    ``dataset_root`` is accepted for API symmetry; parsing does not require it.
    """
    _ = dataset_root
    path_obj = Path(path)
    batch_id = parse_batch_id(path_obj)
    battery_id = parse_battery_id(path_obj)
    protocol_hint = infer_protocol_prefix_from_name(path_obj)
    cycle_id_hint = parse_cycle_id_hint(path_obj)

    parts = [sanitize_identifier(dataset_id, default="dataset")]
    if batch_id:
        parts.append(sanitize_identifier(batch_id))
    if battery_id:
        parts.append(sanitize_identifier(battery_id))
    if len(parts) == 1:
        parts.append(sanitize_identifier(path_obj.stem))
    cell_id = "__".join(parts)

    return CellIdInfo(
        source_file=str(path_obj),
        dataset_id=dataset_id,
        batch_id=batch_id,
        battery_id=battery_id,
        cell_id=cell_id,
        protocol_hint=protocol_hint,
        cycle_id_hint=cycle_id_hint,
    )
