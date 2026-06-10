from __future__ import annotations
import argparse, os, zipfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='D15-P4C pack review zip.')
    p.add_argument('--out-zip', required=True)
    p.add_argument('--paths', nargs='+', required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out_zip); out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for item in args.paths:
            p = Path(item)
            if not p.exists():
                continue
            if p.is_file():
                z.write(p, arcname=p.name)
            else:
                for f in p.rglob('*'):
                    if f.is_file() and f.suffix.lower() in {'.json','.csv','.md','.txt','.log'}:
                        z.write(f, arcname=str(Path(p.name) / f.relative_to(p)))
    print('[D15-P4C pack review] wrote:', out)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
