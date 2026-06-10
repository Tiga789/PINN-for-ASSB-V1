from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Iterable, List


def add_if_exists(z: zipfile.ZipFile, path: Path, arcname: str | None = None) -> None:
    if path.exists() and path.is_file():
        z.write(path, arcname or path.name)


def main() -> int:
    ap = argparse.ArgumentParser(description='D15-P4B pack review zip.')
    ap.add_argument('--config', default='configs/d15_p4b_ready18_generation_config.json')
    ap.add_argument('--preflight-json', required=True)
    ap.add_argument('--generation-dir', required=True)
    ap.add_argument('--audit-dir', required=True)
    ap.add_argument('--scorecard-json', required=True)
    ap.add_argument('--out-zip', default=None)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    out_zip = Path(args.out_zip or cfg['review_zip'])
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    gen_dir = Path(args.generation_dir)
    audit_dir = Path(args.audit_dir)
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        add_if_exists(z, Path(args.preflight_json))
        add_if_exists(z, gen_dir / 'D15_P4B_READY18_RG_GENERATION_REPORT.json')
        add_if_exists(z, gen_dir / 'D15_P4B_READY18_RG_GENERATION_REPORT.csv')
        add_if_exists(z, gen_dir / 'D15_P4B_READY18_RG_GENERATION_ERRORS.json')
        add_if_exists(z, audit_dir / 'radial_gradient_audit_summary.json')
        add_if_exists(z, audit_dir / 'radial_gradient_audit_by_profile.csv')
        add_if_exists(z, audit_dir / 'radial_gradient_audit_by_profile.json')
        add_if_exists(z, Path(args.scorecard_json))
    print('[D15-P4B pack review] wrote:', out_zip)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
