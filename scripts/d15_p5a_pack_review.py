from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser(description='Pack D15-P5A review zip.')
    p.add_argument('--config', default='configs/d15_p5a_all55_existing_model_transfer_config.json')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--out-zip', default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_json(args.config)
    out_dir = Path(args.out_dir or cfg['output_dir'])
    out_zip = Path(args.out_zip or cfg['review_zip'])
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    wanted = [
        'D15_P5A_PREFLIGHT.json',
        'D15_P5A_ALL55_SOFTLABEL_FILE_AUDIT.csv',
        'D15_P5A_MODEL_PREFLIGHT.csv',
        'D15_P5A_PREFLIGHT_READ_ERRORS.json',
        'D15_P5A_FINAL_SCORECARD.json',
        'D15_P5A_MODEL_SCORECARDS.json',
        'D15_P5A_EVAL_ERRORS.json',
        'D15_P5A_METRICS_BY_MODEL_GLOBAL.csv',
        'D15_P5A_METRICS_BY_MODEL_BATCH.csv',
        'D15_P5A_METRICS_BY_MODEL_SEEN_UNSEEN.csv',
        'D15_P5A_METRICS_BY_MODEL_PROFILE.csv',
    ]
    missing: List[str] = []
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name in wanted:
            p = out_dir / name
            if p.exists():
                z.write(p, arcname=name)
            else:
                missing.append(name)
        manifest = {
            'stage': 'D15-P5A review package',
            'out_dir': str(out_dir),
            'missing_files': missing,
            'note': 'This zip intentionally excludes model weights, predictions, and solution_softlabels.npz files.'
        }
        z.writestr('D15_P5A_REVIEW_ZIP_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    print('[D15-P5A pack] wrote:', out_zip)
    if missing:
        print('[D15-P5A pack] missing:', missing)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
