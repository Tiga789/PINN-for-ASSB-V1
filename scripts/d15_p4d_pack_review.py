from __future__ import annotations
import argparse
import zipfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='Pack D15-P4D smoke review files.')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--out-zip', required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_zip = Path(args.out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    patterns = [
        'D15_P4D_SMOKE_FINAL_SCORECARD.json',
        'D15_P4D_CUDA_SMOKE_REPORT.json',
        'D15_P4D_SMOKE_RESOURCE_MONITOR.csv',
        'D15_P4D_SMOKE_GPU_MONITOR.csv',
        'logs/*.json',
        'logs/*.out.log',
        'logs/*.err.log',
    ]
    files = []
    for pat in patterns:
        files.extend(run_dir.glob(pat))
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(set(files)):
            if f.is_file():
                z.write(f, f.relative_to(run_dir))
    print('[D15-P4D smoke pack] wrote:', out_zip)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
