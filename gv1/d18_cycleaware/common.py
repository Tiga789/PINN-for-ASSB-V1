from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


class D18Error(RuntimeError):
    """Base exception for D18-P0/S0/S1 tools."""


class ConfigError(D18Error):
    """Raised when a configuration file is invalid."""


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: str | Path) -> Any:
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ConfigError(f"JSON file not found: {p}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {p}: {exc}") from exc


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=p.parent, encoding=encoding, newline="") as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, p)


def sanitize_json_value(value: Any) -> Any:
    """Recursively convert non-finite values and NumPy scalars to strict JSON values."""
    try:
        import numpy as np  # Local import keeps the helper lightweight for simple verification.
    except Exception:  # pragma: no cover - NumPy is a project dependency.
        np = None  # type: ignore[assignment]

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if np is not None and isinstance(value, np.generic):
        return sanitize_json_value(value.item())
    if np is not None and isinstance(value, np.ndarray):
        return sanitize_json_value(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def dump_json(data: Any, path: str | Path) -> None:
    clean = sanitize_json_value(data)
    atomic_write_text(
        path,
        json.dumps(clean, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False) + "\n",
    )


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path, fieldnames: Sequence[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(str(key))
        fieldnames = keys
    with tempfile.NamedTemporaryFile("w", delete=False, dir=p.parent, encoding="utf-8-sig", newline="") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in fieldnames})
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, p)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(sanitize_json_value(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return value


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json_sha256(data: Any) -> str:
    payload = json.dumps(sanitize_json_value(data), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256_bytes(payload)


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.resolve().as_posix()


def iter_files(
    root: str | Path,
    include_globs: Sequence[str] | None = None,
    exclude_globs: Sequence[str] | None = None,
) -> Iterable[Path]:
    r = Path(root)
    if not r.exists():
        return []
    include_globs = list(include_globs or ["**/*"])
    exclude_globs = list(exclude_globs or [])
    seen: set[Path] = set()
    out: list[Path] = []
    for pattern in include_globs:
        for p in r.glob(pattern):
            if not p.is_file():
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            rel = p.relative_to(r).as_posix()
            if any(Path(rel).match(pattern_ex) for pattern_ex in exclude_globs):
                continue
            if "__pycache__" in p.parts or p.suffix.lower() in {".pyc", ".pyo"}:
                continue
            seen.add(rp)
            out.append(p)
    return sorted(out, key=lambda x: safe_relpath(x, r).lower())


def tree_fingerprint(
    root: str | Path,
    include_globs: Sequence[str] | None = None,
    exclude_globs: Sequence[str] | None = None,
    full_hash_max_bytes: int = 512 * 1024 * 1024,
    max_files: int = 100_000,
) -> dict[str, Any]:
    r = Path(root)
    if not r.exists():
        return {"exists": False, "path": str(r)}
    files = list(iter_files(r, include_globs, exclude_globs))
    truncated = len(files) > max_files
    if truncated:
        files = files[:max_files]
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for p in files:
        st = p.stat()
        total_bytes += st.st_size
        entry: dict[str, Any] = {
            "relative_path": safe_relpath(p, r),
            "size_bytes": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        }
        if st.st_size <= full_hash_max_bytes:
            entry["sha256"] = sha256_file(p)
            entry["hash_mode"] = "full"
        else:
            entry["sha256"] = metadata_fingerprint(p)
            entry["hash_mode"] = "metadata_plus_edges"
        entries.append(entry)
    digest = stable_json_sha256(entries)
    return {
        "exists": True,
        "path": str(r.resolve()),
        "is_dir": r.is_dir(),
        "file_count": len(entries),
        "total_size_bytes": int(total_bytes),
        "truncated": truncated,
        "tree_sha256": digest,
        "entries": entries,
    }


def metadata_fingerprint(path: str | Path, edge_bytes: int = 1024 * 1024) -> str:
    p = Path(path)
    st = p.stat()
    h = hashlib.sha256()
    h.update(str(st.st_size).encode("ascii"))
    h.update(str(st.st_mtime_ns).encode("ascii"))
    with p.open("rb") as f:
        first = f.read(edge_bytes)
        h.update(first)
        if st.st_size > edge_bytes:
            f.seek(max(0, st.st_size - edge_bytes))
            h.update(f.read(edge_bytes))
    return h.hexdigest()


def run_command(command: Sequence[str], cwd: str | Path | None = None, timeout_s: int = 30) -> CommandResult:
    try:
        proc = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
        return CommandResult(list(command), proc.returncode, proc.stdout.strip(), proc.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CommandResult(list(command), 127, "", str(exc))


def collect_git_state(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    commands: dict[str, list[str]] = {
        "is_inside_work_tree": ["git", "rev-parse", "--is-inside-work-tree"],
        "head": ["git", "rev-parse", "HEAD"],
        "short_head": ["git", "rev-parse", "--short=12", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "remote_origin": ["git", "remote", "get-url", "origin"],
        "status_porcelain": ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        "latest_commit": ["git", "log", "-1", "--format=%H%n%cI%n%s"],
        "diff_stat": ["git", "diff", "--stat"],
        "staged_diff_stat": ["git", "diff", "--cached", "--stat"],
    }
    results: dict[str, Any] = {}
    for name, cmd in commands.items():
        result = run_command(cmd, cwd=root)
        results[name] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    inside = results["is_inside_work_tree"]["returncode"] == 0 and results["is_inside_work_tree"]["stdout"] == "true"
    return {
        "collected_at_utc": utc_now_iso(),
        "project_root": str(root.resolve()),
        "git_available": shutil.which("git") is not None,
        "inside_work_tree": inside,
        "commands": results,
    }


def _flatten_context(config: Mapping[str, Any]) -> dict[str, str]:
    context: dict[str, str] = {}
    project = config.get("project", {}) if isinstance(config.get("project"), Mapping) else {}
    paths = config.get("paths", {}) if isinstance(config.get("paths"), Mapping) else {}
    for source in (project, paths):
        for key, value in source.items():
            if isinstance(value, (str, int, float)):
                context[str(key)] = str(value)
    return context


_TEMPLATE_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)\}")


def expand_template(value: str, context: Mapping[str, str]) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value))

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        return context.get(key, match.group(0))

    # Resolve nested config references such as
    # ${d17_split_manifest} -> ${d17_p_root}/split/... -> E:/.../split/... .
    for _ in range(12):
        updated = _TEMPLATE_PATTERN.sub(repl, expanded)
        updated = os.path.expandvars(os.path.expanduser(updated))
        if updated == expanded:
            break
        expanded = updated
    return expanded


def resolve_project_root(config_path: str | Path, config: Mapping[str, Any]) -> Path:
    project = config.get("project")
    if not isinstance(project, Mapping):
        raise ConfigError("config.project must be an object")
    raw = str(project.get("project_root", "."))
    raw = expand_template(raw, _flatten_context(config))
    p = Path(raw)
    if not p.is_absolute():
        # Config is stored in <project>/configs, so ../ is the natural base.
        cfg_parent = Path(config_path).resolve().parent
        candidate_cfg = (cfg_parent / p).resolve()
        candidate_project = (cfg_parent.parent / p).resolve()
        if (candidate_project / "README.md").exists() or (candidate_project / "gv1").exists():
            return candidate_project
        return candidate_cfg
    return p.resolve()


def resolve_config_path(
    raw: str | Path,
    config: Mapping[str, Any],
    project_root: str | Path,
) -> Path:
    context = _flatten_context(config)
    context["project_root"] = str(Path(project_root).resolve())
    value = expand_template(str(raw), context)
    p = Path(value)
    if not p.is_absolute():
        p = Path(project_root) / p
    return p.resolve()


def expand_candidate_paths(
    candidates: Sequence[str],
    config: Mapping[str, Any],
    project_root: str | Path,
) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in candidates:
        p = resolve_config_path(raw, config, project_root)
        text = str(p)
        if any(ch in text for ch in "*?["):
            # Path.glob cannot start with a Windows drive glob reliably on every platform.
            anchor = _glob_anchor(p)
            pattern = str(p)[len(str(anchor)) :].lstrip("/\\")
            matches = anchor.glob(pattern) if anchor.exists() else []
        else:
            matches = [p]
        for match in matches:
            resolved = match.resolve()
            if resolved not in seen:
                seen.add(resolved)
                out.append(resolved)
    return out


def _glob_anchor(path: Path) -> Path:
    parts = list(path.parts)
    safe_parts: list[str] = []
    for part in parts:
        if any(ch in part for ch in "*?["):
            break
        safe_parts.append(part)
    if not safe_parts:
        return Path(".").resolve()
    return Path(*safe_parts)


def ensure_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be an object")
    return value


def ensure_sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(f"{name} must be an array")
    return value


def compact_exception(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}
