#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, os, zipfile
from pathlib import Path

def pfc(v: str) -> Path:
    return Path(str(v).replace('/', os.sep))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/d15_p4a_fix_replay_readiness_config.json')
    ap.add_argument('--out-dir')
    ap.add_argument('--out-zip')
    args = ap.parse_args()
    cfg = json.load(open(args.config, 'r', encoding='utf-8'))
    out = Path(args.out_dir) if args.out_dir else pfc(cfg['output_dir'])
    zpath = Path(args.out_zip) if args.out_zip else pfc(cfg['review_zip'])
    zpath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix.lower() in {'.json', '.csv', '.md', '.txt'}:
                z.write(p, p.relative_to(out).as_posix())
    print('[D15-P4A-fix pack review] wrote:', zpath)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
