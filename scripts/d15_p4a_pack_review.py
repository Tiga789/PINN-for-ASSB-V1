from __future__ import annotations
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

FILES = [
    'D15_P4A_FINAL_SCORECARD.json',
    'D15_P4A_BATCH_COVERAGE_MATRIX.csv',
    'D15_P4A_RAW_CELL_INDEX.csv',
    'D15_P4A_EXISTING_CELL_COVERAGE.csv',
    'D15_P4A_EXISTING_RG_SOFTLABEL_INDEX.csv',
    'D15_P4A_REMAINING32_CELL_MANIFEST.csv',
    'D15_P4A_REPLAY_PROFILE_AUDIT_DEDUP.csv',
    'D15_P4A_P4B_INPUT_MANIFEST.csv',
    'D15_P4A_MISSING_OR_BAD_REPLAY_MANIFEST.csv'
]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--out-zip', required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_zip = Path(args.out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(out_zip, 'w', ZIP_DEFLATED) as z:
        for name in FILES:
            p = out_dir / name
            if p.exists():
                z.write(p, arcname=name)
    print('[D15-P4A pack] wrote:', out_zip)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
