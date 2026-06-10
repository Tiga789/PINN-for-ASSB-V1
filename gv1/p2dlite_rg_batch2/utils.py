from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


def pathify(p: str | Path) -> Path:
    return Path(str(p).replace('\\\\', '/'))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(obj: Mapping[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(_jsonify(obj), f, indent=2, ensure_ascii=False)


def _jsonify(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return _jsonify(x.item())
        return [_jsonify(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        val = float(x)
        return None if not math.isfinite(val) else val
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    if isinstance(x, Path):
        return str(x)
    return x


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path, fieldnames: Optional[Sequence[str]] = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys or ['empty']
    with open(p, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow({k: _csv_val(row.get(k, '')) for k in fieldnames})


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def _csv_val(v: Any) -> Any:
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(_jsonify(v), ensure_ascii=False)
    if isinstance(v, np.ndarray):
        return json.dumps(_jsonify(v), ensure_ascii=False)
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, float) and not math.isfinite(v):
        return ''
    return v


def parse_battery_id(name: str) -> Optional[int]:
    s = str(name)
    pats = [r'battery[-_ ]?(\d+)', r'cell[-_ ]?(\d+)', r'bat[-_ ]?(\d+)', r'(?:^|[_\- ])(\d{1,2})(?:\D|$)']
    for pat in pats:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None


def scalar_string(x: Any, default: str = '') -> str:
    try:
        arr = np.asarray(x)
        if arr.ndim == 0:
            return str(arr.item())
        if arr.size == 1:
            return str(arr.reshape(-1)[0])
    except Exception:
        pass
    return default


def discover_npz(root: str | Path, filename: str = 'solution_softlabels.npz') -> List[Path]:
    r = Path(root)
    if not r.exists():
        return []
    return sorted([p for p in r.rglob(filename) if p.is_file()])


def safe_status(pass_bool: bool, review_bool: bool = False) -> str:
    if not pass_bool:
        return 'FAIL'
    return 'REVIEW' if review_bool else 'PASS'


def format_profile_name(batch: str, protocol: str, battery_id: int | str) -> str:
    bid = str(battery_id).strip()
    if bid.lower().startswith('battery'):
        bid = parse_battery_id(bid) or bid
    return f'{batch}_{protocol}_battery-{bid}'
