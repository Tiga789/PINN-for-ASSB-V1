from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='Pack lightweight D15-P3B review zip.')
    p.add_argument('--repair-dir', required=True)
    p.add_argument('--scorecard-json', required=True)
    p.add_argument('--out-zip', required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repair_dir = Path(args.repair_dir)
    out_zip = Path(args.out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    candidates = [
        Path(args.scorecard_json),
        repair_dir / 'D15_P3B_BOUNDARY_REPAIR_SUMMARY.json',
        repair_dir / 'D15_P3B_BOUNDARY_REPAIR_BY_PROFILE.csv',
        repair_dir / 'D15_P3B_BOUNDARY_REPAIR_BY_PROFILE.json',
        repair_dir / 'D15_P3B_TOP_RAW_THETA_OUTSIDE_POINTS.csv',
        repair_dir / 'D15_P3B_TOP_PROJECTED_THETA_ERROR_POINTS.csv',
    ]
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for p in candidates:
            if p.exists() and p.is_file():
                zf.write(p, arcname=p.name)
        # Also include any JSON/CSV directly under repair dir for robustness, excluding predictions.
        for p in sorted(repair_dir.glob('*.json')) + sorted(repair_dir.glob('*.csv')):
            if p.name not in zf.namelist():
                zf.write(p, arcname=p.name)
    print('[D15-P3B pack review] wrote:', out_zip)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
