from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Iterable


def add_if_exists(z: zipfile.ZipFile, path: Path, arc: str | None = None) -> None:
    if path.exists() and path.is_file():
        z.write(path, arc or path.name)


def add_tree_selected(z: zipfile.ZipFile, root: Path, prefix: str, suffixes: Iterable[str]) -> None:
    if not root.exists():
        return
    suffixes = tuple(s.lower() for s in suffixes)
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in suffixes:
            z.write(p, f'{prefix}/{p.relative_to(root).as_posix()}')


def main() -> int:
    ap = argparse.ArgumentParser(description='Pack D15-P4D Batch-5_battery-8 targeted fix review zip.')
    ap.add_argument('--run-dir', default='E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p4d_batch5_battery8_targeted_fix')
    ap.add_argument('--out-zip', default='E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p4d_batch5_battery8_targeted_fix_review.zip')
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    out_zip = Path(args.out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        add_if_exists(z, run_dir / 'D15_P4D_BATCH5_BATTERY8_TARGETED_FIX_SUMMARY.json')
        add_if_exists(z, run_dir / 'D15_P4D_BATCH5_BATTERY8_TARGETED_FIX_CANDIDATES.csv')
        add_tree_selected(z, run_dir / 'selected_candidate' / 'audit', 'selected_candidate/audit', ['.json', '.csv'])
        add_tree_selected(z, run_dir / 'logs', 'logs', ['.log'])
        # Add small status JSON files, but never include solution_softlabels.npz.
        add_tree_selected(z, run_dir / 'candidates', 'candidates_status_and_audit', ['.json', '.csv'])
    print('[D15-P4D targeted pack] wrote:', out_zip)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
