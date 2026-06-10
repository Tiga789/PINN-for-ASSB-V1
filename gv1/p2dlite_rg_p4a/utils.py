from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def write_csv(path: str | Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    fieldnames.append(k)
                    seen.add(k)
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k, '')) for k in fieldnames})


def _csv_value(v: Any) -> Any:
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v


def parse_cell_identity_from_text(text: str, parent_batch: Optional[int] = None) -> Tuple[Optional[int], Optional[int], str, str]:
    s = str(text).replace('\\', '/')
    batch_id = parent_batch
    m = re.search(r'Batch[-_ ]?(\d+)', s, re.IGNORECASE)
    if m:
        batch_id = int(m.group(1))
    protocol = 'unknown'
    if re.search(r'R2\.5', s, re.IGNORECASE):
        protocol = 'R2.5'
        if batch_id is None:
            batch_id = 3
    elif re.search(r'R3', s, re.IGNORECASE):
        protocol = 'R3'
        if batch_id is None:
            batch_id = 4
    elif re.search(r'3C', s, re.IGNORECASE):
        protocol = '3C'
        if batch_id is None:
            batch_id = 2
    elif re.search(r'2C', s, re.IGNORECASE):
        protocol = '2C'
        if batch_id is None:
            batch_id = 1
    elif batch_id == 5:
        protocol = 'random_walk'
    elif batch_id == 6:
        protocol = 'GEO'
    matches = re.findall(r'battery[-_ ]?(\d+)', s, re.IGNORECASE)
    battery_id = int(matches[-1]) if matches else None
    canonical = f'Batch-{batch_id}_battery-{battery_id}' if batch_id is not None and battery_id is not None else ''
    return batch_id, battery_id, protocol, canonical


def batch_protocol(batch_id: Optional[int], fallback: str = 'unknown') -> str:
    return {1: '2C', 2: '3C', 3: 'R2.5', 4: 'R3', 5: 'random_walk', 6: 'GEO'}.get(batch_id, fallback)


def discover_raw_mat_cells(dataset_root: str | Path) -> List[Dict[str, Any]]:
    root = Path(dataset_root)
    rows: List[Dict[str, Any]] = []
    for batch_dir in sorted(root.glob('Batch-*')):
        if not batch_dir.is_dir():
            continue
        m = re.search(r'Batch[-_ ]?(\d+)', batch_dir.name, re.IGNORECASE)
        parent_batch = int(m.group(1)) if m else None
        for fp in sorted(batch_dir.glob('*.mat')):
            batch_id, battery_id, protocol, canonical = parse_cell_identity_from_text(str(fp), parent_batch=parent_batch)
            rows.append({
                'canonical_cell_id': canonical,
                'batch_id': batch_id if batch_id is not None else '',
                'battery_id': battery_id if battery_id is not None else '',
                'protocol_inferred': batch_protocol(batch_id, protocol),
                'raw_mat_path': str(fp),
                'raw_mat_exists': fp.exists(),
                'raw_mat_size_bytes': fp.stat().st_size if fp.exists() else '',
            })
    dedup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = row.get('canonical_cell_id') or row.get('raw_mat_path')
        old = dedup.get(key)
        if old is None or int(row.get('raw_mat_size_bytes') or 0) > int(old.get('raw_mat_size_bytes') or 0):
            dedup[key] = row
    out = list(dedup.values())
    out.sort(key=lambda r: (int(r.get('batch_id') or 999), int(r.get('battery_id') or 999)))
    return out
