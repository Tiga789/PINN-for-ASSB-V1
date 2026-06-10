from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='Pack D15-P3C lightweight review zip.')
    p.add_argument('--cache-root', required=True)
    p.add_argument('--out-zip', required=True)
    p.add_argument('--extra', action='append', default=[])
    return p.parse_args()


def _add_existing(zf: zipfile.ZipFile, path: Path, base: Path) -> None:
    if not path.exists():
        return
    if path.is_file():
        if path.suffix.lower() in {'.json', '.csv', '.md', '.txt', '.log'}:
            zf.write(path, path.relative_to(base) if path.is_relative_to(base) else path.name)
        return
    for p in path.rglob('*'):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {'.json', '.csv', '.md', '.txt', '.log'}:
            continue
        # Avoid very large logs if any.
        try:
            if p.stat().st_size > 20_000_000:
                continue
        except Exception:
            pass
        zf.write(p, p.relative_to(base) if p.is_relative_to(base) else p.name)


def main() -> int:
    args = parse_args()
    base = Path(args.cache_root)
    out_zip = Path(args.out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    targets = [
        base / 'xjtu_d15_p3c_batch2_15cell_applicability_scorecard',
        base / 'xjtu_softlabels_p2dlite_rg_v1_d15p3c_batch2_15cell' / 'D15_P3C_BATCH2_15CELL_RG_GENERATION_REPORT.json',
        base / 'xjtu_softlabels_p2dlite_rg_v1_d15p3c_batch2_15cell' / 'D15_P3C_BATCH2_15CELL_RG_GENERATION_REPORT.csv',
        base / 'xjtu_softlabels_p2dlite_rg_v1_d15p3c_batch2_15cell' / 'profiles',
        base / 'xjtu_d15_p3c_batch2_15cell_radial_audit',
        base / 'xjtu_d15_p3c_batch2_15cell_rg_nn_benchmark' / 'D15_P1_FINAL_SCORECARD.json',
        base / 'xjtu_d15_p3c_batch2_15cell_rg_nn_benchmark' / 'D15_P1_TRAINING_SUMMARY.json',
        base / 'xjtu_d15_p3c_batch2_15cell_rg_nn_benchmark' / 'D15_P1_DATASET_SAMPLING_SUMMARY.json',
        base / 'xjtu_d15_p3c_batch2_15cell_rg_nn_benchmark' / 'training_history.csv',
        base / 'xjtu_d15_p3c_batch2_15cell_rg_nn_benchmark' / 'eval_full_profiles' / 'D15_P1_EVAL_SUMMARY.json',
        base / 'xjtu_d15_p3c_batch2_15cell_rg_nn_benchmark' / 'eval_full_profiles' / 'D15_P1_METRICS_BY_PROFILE.csv',
        base / 'xjtu_d15_p3c_batch2_15cell_boundary_projection_repair',
    ]
    targets.extend(Path(x) for x in args.extra)
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for t in targets:
            _add_existing(zf, t, base)
    print('[D15-P3C pack review] wrote:', out_zip)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
