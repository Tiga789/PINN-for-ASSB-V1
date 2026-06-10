from __future__ import annotations
import argparse
import zipfile
from pathlib import Path


def add_file(z: zipfile.ZipFile, path: Path, base: Path) -> None:
    if path.exists() and path.is_file():
        z.write(path, path.relative_to(base).as_posix())


def add_dir_filtered(z: zipfile.ZipFile, root: Path, base: Path, exts=('.json', '.csv', '.log', '.txt')) -> None:
    if not root.exists():
        return
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            z.write(p, p.relative_to(base).as_posix())


def main() -> int:
    ap = argparse.ArgumentParser(description='D15-P4D full: pack review zip without large npz/model files.')
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--softlabel-dir', required=True)
    ap.add_argument('--audit-dir', required=True)
    ap.add_argument('--scorecard-dir', required=True)
    ap.add_argument('--out-zip', required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    soft = Path(args.softlabel_dir)
    audit = Path(args.audit_dir)
    score = Path(args.scorecard_dir)
    out = Path(args.out_zip)
    out.parent.mkdir(parents=True, exist_ok=True)
    base = run_dir.parent
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        add_dir_filtered(z, run_dir, base)
        add_file(z, soft / 'D15_P4D_BATCH56_REMAINING14_RG_GENERATION_REPORT.json', base)
        add_file(z, soft / 'D15_P4D_BATCH56_REMAINING14_RG_GENERATION_REPORT.csv', base)
        add_dir_filtered(z, audit, base)
        add_dir_filtered(z, score, base)
        # Include only per-cell soft_label_summary.json files, not solution_softlabels.npz.
        if soft.exists():
            for p in soft.rglob('soft_label_summary.json'):
                z.write(p, p.relative_to(base).as_posix())
    print('[D15-P4D pack review] wrote:', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
