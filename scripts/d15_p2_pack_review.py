from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='Pack small D15-P2 review artifacts without model weights or full prediction arrays.')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--eval-dir', required=True)
    p.add_argument('--audit-dir', required=True)
    p.add_argument('--out-zip', required=True)
    return p.parse_args()


def add_if_exists(z: zipfile.ZipFile, path: Path, arcname: str | None = None) -> None:
    if path.exists() and path.is_file():
        z.write(path, arcname or path.name)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    eval_dir = Path(args.eval_dir)
    audit_dir = Path(args.audit_dir)
    out_zip = Path(args.out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name in [
            'D15_P2_PREFLIGHT.json',
            'D15_P2_TRAINING_SUMMARY.json',
            'D15_P2_DATASET_SAMPLING_SUMMARY.json',
            'training_history.csv',
            'D15_P2_FINAL_SCORECARD.json',
        ]:
            add_if_exists(z, run_dir / name, f'run/{name}')
        # Also keep original P1-name aliases if present because they are useful for debugging wrapper runs.
        for name in ['D15_P1_TRAINING_SUMMARY.json', 'D15_P1_DATASET_SAMPLING_SUMMARY.json']:
            add_if_exists(z, run_dir / name, f'run/{name}')
        for name in [
            'D15_P2_EVAL_SUMMARY.json',
            'D15_P2_METRICS_BY_PROFILE.csv',
            'D15_P2_METRICS_BY_PROFILE.json',
            'D15_P1_EVAL_SUMMARY.json',
            'D15_P1_METRICS_BY_PROFILE.csv',
        ]:
            add_if_exists(z, eval_dir / name, f'eval/{name}')
        for name in [
            'D15_P2_PRECISION_AUDIT_SUMMARY.json',
            'D15_P2_PRECISION_AUDIT_BY_PROFILE.csv',
            'D15_P2_TOPK_ERROR_WINDOWS.csv',
            'D15_P2_CYCLE_LEVEL_AUDIT.csv',
        ]:
            add_if_exists(z, audit_dir / name, f'audit/{name}')
    print('[D15-P2 pack] wrote:', out_zip)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
