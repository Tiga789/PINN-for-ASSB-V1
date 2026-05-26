from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError('Reading YAML manifests requires PyYAML. Install pyyaml or use JSON manifests.') from exc
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data or {}


def load_manifest(path: str | Path | None) -> dict[str, Any]:
    """Load YAML or JSON manifest. Missing path returns empty dict."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Manifest does not exist: {p}')
    if p.suffix.lower() in {'.yaml', '.yml'}:
        return _read_yaml(p)
    if p.suffix.lower() == '.json':
        return json.loads(p.read_text(encoding='utf-8'))
    raise ValueError(f'Unsupported manifest suffix: {p.suffix}')


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_update(dict(out[k]), v)
        elif v is not None:
            out[k] = v
    return out


def merge_cli_overrides(manifest: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Merge simple CLI overrides into top-level manifest."""
    return deep_update(manifest, {k: v for k, v in overrides.items() if v is not None})


def write_resolved_manifest(manifest: Mapping[str, Any], output_path: str | Path) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(manifest), ensure_ascii=False, indent=2), encoding='utf-8')
