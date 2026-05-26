"""Generic recursive dataset discovery for GV1.

The discovery layer only finds files and attaches path-level metadata.  It does
not read battery data arrays; reading remains in ``gv1.io``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_FILE_PATTERNS = ("*.mat", "*.csv", "*.parquet")
DEFAULT_EXCLUDE_DIRS = (
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    "CacheGV1",
    "DataGV1",
    "ModelGV1",
    "EvalGV1",
)


def _norm_batch(value: str) -> str:
    return value.strip().replace("_", "-").lower()


def _mtime_iso(path: Path) -> str:
    try:
        dt = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return dt.isoformat()
    except Exception:
        return ""


def _is_under_included_batch(path: Path, root: Path, include_batches: Sequence[str] | None) -> bool:
    if not include_batches:
        return True
    wanted = {_norm_batch(x) for x in include_batches}
    try:
        rel_parts = path.relative_to(root).parts
    except Exception:
        rel_parts = path.parts
    return any(_norm_batch(p) in wanted for p in rel_parts)


def _is_excluded(path: Path, exclude_dirs: Sequence[str]) -> bool:
    excluded = set(exclude_dirs)
    return any(part in excluded for part in path.parts)


def _matches_patterns(path: Path, root: Path, patterns: Sequence[str]) -> bool:
    try:
        rel = path.relative_to(root).as_posix()
    except Exception:
        rel = path.as_posix()
    name = path.name
    for pat in patterns:
        if fnmatch(name.lower(), pat.lower()) or fnmatch(rel.lower(), pat.lower()):
            return True
    return False


@dataclass(frozen=True)
class DiscoveredFile:
    dataset_root: str
    source_file: str
    relative_path: str
    source_format: str
    file_size_bytes: int
    mtime_iso: str

    def to_dict(self) -> dict:
        return asdict(self)


def discover_files(
    dataset_root: str | Path,
    *,
    file_patterns: Sequence[str] = DEFAULT_FILE_PATTERNS,
    include_batches: Sequence[str] | None = None,
    exclude_dirs: Sequence[str] = DEFAULT_EXCLUDE_DIRS,
    recursive: bool = True,
    max_files: int | None = None,
) -> list[DiscoveredFile]:
    """Discover candidate battery data files under ``dataset_root``.

    Parameters are intentionally generic; use manifests or adapters for
    dataset-specific behavior.
    """
    root = Path(dataset_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"dataset_root is not a directory: {root}")

    iterator: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    out: list[DiscoveredFile] = []
    for path in iterator:
        if max_files is not None and len(out) >= max_files:
            break
        if not path.is_file():
            continue
        if _is_excluded(path, exclude_dirs):
            continue
        if not _is_under_included_batch(path, root, include_batches):
            continue
        if not _matches_patterns(path, root, file_patterns):
            continue
        try:
            size = path.stat().st_size
        except Exception:
            size = -1
        try:
            rel = path.relative_to(root).as_posix()
        except Exception:
            rel = path.name
        fmt = path.suffix.lower().lstrip(".") or "unknown"
        out.append(
            DiscoveredFile(
                dataset_root=str(root),
                source_file=str(path),
                relative_path=rel,
                source_format=fmt,
                file_size_bytes=int(size),
                mtime_iso=_mtime_iso(path),
            )
        )
    out.sort(key=lambda x: x.relative_path)
    return out


def summarize_discovery(files: Sequence[DiscoveredFile]) -> dict:
    by_format: dict[str, int] = {}
    total_bytes = 0
    for f in files:
        by_format[f.source_format] = by_format.get(f.source_format, 0) + 1
        if f.file_size_bytes > 0:
            total_bytes += f.file_size_bytes
    return {
        "file_count": len(files),
        "by_format": dict(sorted(by_format.items())),
        "total_size_bytes": total_bytes,
    }
