#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.pipeline.manifest import load_manifest, merge_cli_overrides, write_resolved_manifest
from gv1.pipeline.data_loader import iter_standard_tables_from_index
from gv1.measured_replay.profile_builder import build_replay_profile
from gv1.measured_replay.replay_audit import audit_replay_profile
from gv1.measured_replay.capacity_integrator import build_cycle_integrals


def _get(manifest: dict, key: str, default=None):
    cur = manifest
    for part in key.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _safe_name(value: object) -> str:
    s = str(value if value not in (None, '', 'nan') else 'unknown')
    s = re.sub(r'[^A-Za-z0-9_.-]+', '_', s).strip('_')
    return s or 'unknown'


def _capacity_normalized_baseline(profile, q_ref_Ah: float | None) -> dict[str, np.ndarray]:
    """Lightweight cbar-like baseline for GV1 measured-current replay plumbing.

    This is not the final electrochemical soft label.  It allows the downstream
    pipeline to test I(t)-conditioned replay before the full effective-SPM
    numerical solver is connected.
    """
    q_net = profile.q_net_Ah if profile.q_net_Ah is not None else np.zeros_like(profile.t_s)
    if q_ref_Ah is None or not np.isfinite(q_ref_Ah) or q_ref_Ah <= 0:
        q_ref_Ah = max(float(np.nanmax(profile.throughput_Ah)) if profile.throughput_Ah is not None else 1.0, 1e-12)
    delta = q_net / float(q_ref_Ah)
    return {
        'cbar_a_norm_replay': np.clip(0.5 + delta, 0.0, 1.0),
        'cbar_c_norm_replay': np.clip(0.5 - delta, 0.0, 1.0),
        'Q_ref_Ah_replay': np.asarray([q_ref_Ah], dtype=float),
    }


def _save_npz(path: Path, arrays: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {}
    for k, v in arrays.items():
        arr = np.asarray(v)
        if arr.dtype == object:
            arr = arr.astype(str)
        clean[k] = arr
    np.savez_compressed(path, **clean)


def _profile_group_columns(df: pd.DataFrame) -> list[str]:
    # One .mat/parquet file usually corresponds to one cell. Group by source_file
    # first so multiple cells/batches never become one artificial time profile.
    for cols in [
        ['source_file'],
        ['dataset_id', 'batch_id', 'battery_id', 'cell_id'],
        ['dataset_id', 'batch_id', 'battery_id'],
    ]:
        if all(c in df.columns for c in cols):
            return cols
    return []


def _write_root_smoke_copy(out_dir: Path, arrays: dict[str, object], profile, audit) -> None:
    _save_npz(out_dir / 'solution_replay_profile.npz', arrays)
    (out_dir / 'profile_summary.json').write_text(json.dumps(profile.summary(), ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'replay_audit.json').write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='Build GV1 measured-current replay profile/soft-label input bundles from a dataset index.')
    ap.add_argument('--manifest', default=None, help='YAML/JSON manifest. CLI args override it.')
    ap.add_argument('--dataset_index_csv', default=None)
    ap.add_argument('--dataset_root', default=None)
    ap.add_argument('--adapter', default=None, help='xjtu or generic')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--max_files', type=int, default=None)
    ap.add_argument('--default_temperature_C', type=float, default=None)
    ap.add_argument('--q_ref_Ah', type=float, default=None)
    ap.add_argument('--write_standard_csv', action='store_true', help='Write per-profile standard_table.csv. Avoid for full XJTU runs.')
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    manifest = merge_cli_overrides(
        manifest,
        dataset_index_csv=args.dataset_index_csv,
        dataset_root=args.dataset_root,
        adapter=args.adapter,
        output_dir=args.output_dir,
        max_files=args.max_files,
        default_temperature_C=args.default_temperature_C,
        q_ref_Ah=args.q_ref_Ah,
    )

    index_csv = manifest.get('dataset_index_csv') or _get(manifest, 'dataset.index_csv')
    if not index_csv:
        raise ValueError('dataset_index_csv is required. Build it with scripts/gv1_build_dataset_index.py first.')

    out_dir = Path(manifest.get('output_dir') or args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = manifest.get('adapter') or _get(manifest, 'dataset.adapter', 'xjtu')
    default_temp = float(manifest.get('default_temperature_C') or _get(manifest, 'dataset.default_temperature_C', 25.0))
    max_files = manifest.get('max_files')
    max_files = int(max_files) if max_files not in (None, '') else None
    q_ref = manifest.get('q_ref_Ah')
    q_ref = float(q_ref) if q_ref not in (None, '') else None

    manifest_rows: list[dict[str, object]] = []
    cycle_summaries: list[pd.DataFrame] = []
    first_npz: Path | None = None
    profile_index = 0

    # Stream one source file at a time. This avoids concatenating the 24 XJTU
    # multi-million-row files into one huge DataFrame.
    for file_index, (row, df) in enumerate(
        iter_standard_tables_from_index(
            index_csv,
            adapter=adapter,
            dataset_root=manifest.get('dataset_root') or _get(manifest, 'dataset.root'),
            max_files=max_files,
            default_temperature_C=default_temp,
        ),
        start=1,
    ):
        df = df.reset_index(drop=True)
        group_cols = _profile_group_columns(df)
        if group_cols:
            groups = df.groupby(group_cols, dropna=False, sort=False)
        else:
            groups = [(('all',), df)]

        for keys, g in groups:
            profile_index += 1
            if not isinstance(keys, tuple):
                keys = (keys,)
            meta = dict(zip(group_cols, keys)) if group_cols else {'profile_id': 'all'}
            cell = _safe_name(meta.get('cell_id') or meta.get('cell_uid') or meta.get('battery_id') or row.get('battery_id') or f'profile_{profile_index:04d}')
            source_name = _safe_name(Path(str(meta.get('source_file') or row.get('source_file') or f'profile_{profile_index:04d}')).stem)
            profile_id = f'{profile_index:04d}_{cell}_{source_name}'
            profile_dir = out_dir / 'profiles' / profile_id

            if args.write_standard_csv:
                profile_dir.mkdir(parents=True, exist_ok=True)
                g.to_csv(profile_dir / 'standard_table.csv', index=False, encoding='utf-8-sig')

            profile = build_replay_profile(g)
            arrays = profile.to_npz_dict()
            arrays.update(_capacity_normalized_baseline(profile, q_ref))
            npz_path = profile_dir / 'solution_replay_profile.npz'
            _save_npz(npz_path, arrays)

            audit = audit_replay_profile(profile)
            profile_dir.mkdir(parents=True, exist_ok=True)
            (profile_dir / 'profile_summary.json').write_text(json.dumps(profile.summary(), ensure_ascii=False, indent=2), encoding='utf-8')
            (profile_dir / 'replay_audit.json').write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')

            manifest_rows.append({
                'profile_id': profile_id,
                'npz_path': str(npz_path),
                'summary_json': str(profile_dir / 'profile_summary.json'),
                'audit_json': str(profile_dir / 'replay_audit.json'),
                'ok': audit.ok,
                'n_points': int(len(profile.t_s)),
                **{k: str(v) for k, v in meta.items()},
            })
            if first_npz is None:
                first_npz = npz_path
                # Preserve old smoke behavior only for single-file runs.
                if max_files is not None and max_files <= 1:
                    _write_root_smoke_copy(out_dir, arrays, profile, audit)

        try:
            cycle_summary = build_cycle_integrals(df)
            cycle_summary.insert(0, 'source_index', file_index)
            cycle_summaries.append(cycle_summary)
        except Exception as exc:
            (out_dir / f'cycle_integrals_source_{file_index:04d}.failed.txt').write_text(str(exc), encoding='utf-8')

        # Release the large file before reading the next one.
        del df
        gc.collect()

    if not manifest_rows:
        raise ValueError('No replay profiles were generated')

    pd.DataFrame(manifest_rows).to_csv(out_dir / 'profile_manifest.csv', index=False, encoding='utf-8-sig')
    if cycle_summaries:
        pd.concat(cycle_summaries, ignore_index=True, sort=False).to_csv(out_dir / 'cycle_integrals.csv', index=False, encoding='utf-8-sig')
    write_resolved_manifest(manifest, out_dir / 'resolved_manifest.json')

    ok_all = bool(manifest_rows) and all(bool(r['ok']) for r in manifest_rows)
    print(json.dumps({
        'ok': ok_all,
        'output_dir': str(out_dir),
        'profile_count': len(manifest_rows),
        'first_npz': str(first_npz) if first_npz else None,
        'streaming_mode': True,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
