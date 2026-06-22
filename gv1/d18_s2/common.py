from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class D18S2Error(RuntimeError):
    """Base exception for D18-S2 preflight and micro-smoke."""


class ConfigError(D18S2Error):
    """Raised when configuration or prerequisite data are invalid."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_json_value(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None  # type: ignore[assignment]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if np is not None and isinstance(value, np.generic):
        return sanitize_json_value(value.item())
    if np is not None and isinstance(value, np.ndarray):
        return sanitize_json_value(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_json_value(v) for v in value]
    return str(value)


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
        temp = Path(tmp.name)
    os.replace(temp, p)


def dump_json(data: Any, path: str | Path) -> None:
    clean = sanitize_json_value(data)
    atomic_write_text(path, json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(sanitize_json_value(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return value


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path, fieldnames: Sequence[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: set[str] = set()
        names: list[str] = []
        for row in rows:
            for key in row:
                key = str(key)
                if key not in seen:
                    seen.add(key)
                    names.append(key)
        fieldnames = names or ["status"]
    with tempfile.NamedTemporaryFile("w", delete=False, dir=p.parent, encoding="utf-8-sig", newline="") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in fieldnames})
        temp = Path(tmp.name)
    os.replace(temp, p)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"CSV file not found: {p}")
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def expand_template(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        out = value
        for _ in range(20):
            prior = out
            for key, replacement in context.items():
                out = out.replace("${" + str(key) + "}", str(replacement))
            out = os.path.expandvars(os.path.expanduser(out))
            if out == prior:
                break
        return out
    if isinstance(value, Mapping):
        return {k: expand_template(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_template(v, context) for v in value]
    return value


def resolve_config(path: str | Path, *, project_root: str | Path | None = None) -> dict[str, Any]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise ConfigError("Top-level config must be an object")
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    context: dict[str, Any] = {"project_root": str(root)}
    paths = raw.get("paths", {})
    if isinstance(paths, Mapping):
        for _ in range(10):
            changed = False
            for key, value in paths.items():
                resolved = expand_template(value, {**context, **paths})
                if paths.get(key) != resolved:
                    paths[key] = resolved
                    changed = True
                context[str(key)] = resolved
            if not changed:
                break
    raw["paths"] = paths
    return expand_template(raw, {**context, **paths})


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_path_text(path: str | Path) -> str:
    return str(path).replace("/", "\\").rstrip("\\").lower()


def exact_numeric_token(text: str, prefix: str) -> int | None:
    match = re.search(rf"(?<![A-Za-z0-9]){re.escape(prefix)}-(\d+)(?!\d)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def choose_device(requested: str = "auto") -> str:
    import torch
    value = requested.strip().lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise ConfigError("CUDA was requested but torch.cuda.is_available() is false")
    if value not in {"cpu", "cuda"}:
        raise ConfigError(f"Unsupported device: {requested}")
    return value


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def flatten_records(value: Any) -> list[dict[str, Any]]:
    """Return dict records from common manifest layouts without guessing scalar values."""
    if isinstance(value, list):
        return [dict(x) for x in value if isinstance(x, Mapping)]
    if isinstance(value, Mapping):
        for key in ("records", "profiles", "rows", "items"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                return [dict(x) for x in candidate if isinstance(x, Mapping)]
    raise ConfigError("Could not locate a record list in manifest")


def safe_unlink(path: str | Path) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
