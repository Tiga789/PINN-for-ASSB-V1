from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_nn.data import build_dataset
from gv1.p2dlite_rg_nn.metrics import compute_rg_metrics, thresholds_status
from gv1.p2dlite_rg_nn.model import build_model
from gv1.p2dlite_rg_nn.train_eval import evaluate_one_profile, predict_numpy
from gv1.p2dlite_rg_nn.utils import discover_npz, ensure_clean_or_allowed, load_json, write_csv, write_json


def parse_args():
    p = argparse.ArgumentParser(description='D15-P1 evaluate closed-set NN on full P2Dlite-RG profiles.')
    p.add_argument('--softlabel-dir', required=True)
    p.add_argument('--model-dir', required=True, help='D15-P1 output directory containing model/best_with_state.pt, or the model subdirectory itself.')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--config', default='configs/d15_p1_nn_smoke_config.json')
    p.add_argument('--filename', default='solution_softlabels.npz')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=65536)
    p.add_argument('--eval-stride', type=int, default=None)
    p.add_argument('--allow-overwrite', action='store_true')
    p.add_argument('--save-prediction-npz', action='store_true')
    return p.parse_args()


def _device(name: str):
    import torch
    if name == 'auto' or not name:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def _model_file(model_dir: Path) -> Path:
    if (model_dir / 'model' / 'best_with_state.pt').exists():
        return model_dir / 'model' / 'best_with_state.pt'
    if (model_dir / 'best_with_state.pt').exists():
        return model_dir / 'best_with_state.pt'
    raise FileNotFoundError(f'Could not find best_with_state.pt under {model_dir} or {model_dir / "model"}')


def main() -> int:
    args = parse_args()
    import torch
    cfg = load_json(args.config)
    out_dir = ensure_clean_or_allowed(args.out_dir, allow_overwrite=args.allow_overwrite)
    thresholds = cfg.get('scorecard_thresholds', {})
    mf = _model_file(Path(args.model_dir))
    device = _device(args.device)
    ck = torch.load(mf, map_location=device, weights_only=False)
    state = ck['state']
    model = build_model(int(state['input_dim']), int(state['output_dim']), state['model_config']).to(device)
    model.load_state_dict(ck['model_state_dict'])
    root = Path(args.softlabel_dir)
    files = discover_npz(root, filename=args.filename)
    profile_ids = list(state.get('profile_ids', []))
    profile_count = len(profile_ids)
    if len(files) != profile_count:
        print(f'[D15-P1 eval] WARNING: found {len(files)} files but model has profile_count={profile_count}', flush=True)
    # Map path/profile id to original index. Fall back to sorted order.
    rows: List[Dict[str, Any]] = []
    all_true = []
    all_pred = []
    eval_stride = int(args.eval_stride if args.eval_stride is not None else cfg.get('data', {}).get('eval_stride', 1))
    from gv1.p2dlite_rg_nn.data import profile_id_from_path, load_profile_arrays, build_features, build_targets
    for default_i, npz_path in enumerate(files):
        pid = profile_id_from_path(npz_path, root)
        try:
            i = profile_ids.index(pid)
        except ValueError:
            i = default_i
        save_path = None
        if args.save_prediction_npz:
            safe = pid.replace('/', '__').replace('\\', '__').replace(':', '_')
            save_path = out_dir / 'predictions' / f'{safe}_prediction.npz'
        row = evaluate_one_profile(npz_path, root, i, profile_count, model, state, device, batch_size=args.batch_size, eval_stride=eval_stride, save_prediction_path=save_path)
        rows.append(row)
        # To avoid reading predictions twice, re-run a lightweight prediction for global aggregation only on stride data.
        prof = load_profile_arrays(npz_path, root)
        X, _ = build_features(prof, i, profile_count, include_profile_onehot=bool(state.get('include_profile_onehot', True)))
        Y, _, _ = build_targets(prof)
        stride = max(1, eval_stride)
        X_eval = X[::stride]
        Y_eval = Y[::stride]
        Y_pred = predict_numpy(model, X_eval, np.asarray(state['x_mean'], dtype=np.float32), np.asarray(state['x_std'], dtype=np.float32), np.asarray(state['y_mean'], dtype=np.float32), np.asarray(state['y_std'], dtype=np.float32), device, batch_size=args.batch_size)
        all_true.append(Y_eval)
        all_pred.append(Y_pred)
        print(f'[D15-P1 eval] {pid}: phis_c_mae={row.get("phis_c_mae"):.6g} theta_a_mae={row.get("theta_a_mae"):.6g} theta_c_mae={row.get("theta_c_mae"):.6g}', flush=True)
    YT = np.concatenate(all_true, axis=0)
    YP = np.concatenate(all_pred, axis=0)
    global_metrics = compute_rg_metrics(YT, YP, state['target_slices'])
    score = thresholds_status(dict(global_metrics), thresholds)
    summary = {
        'stage': 'D15-P1 closed-set NN full-profile evaluation',
        'softlabel_dir': str(root),
        'model_file': str(mf),
        'out_dir': str(out_dir),
        'profile_count': len(rows),
        'eval_stride': eval_stride,
        'global_metrics': global_metrics,
        'scorecard': score,
        'overall_status': score['overall_status'],
        'notes': cfg.get('audit_notes', []),
    }
    write_csv(rows, out_dir / 'D15_P1_METRICS_BY_PROFILE.csv')
    write_json(rows, out_dir / 'D15_P1_METRICS_BY_PROFILE.json')
    write_json(summary, out_dir / 'D15_P1_EVAL_SUMMARY.json')
    print('[D15-P1 eval] overall_status:', summary['overall_status'])
    print('[D15-P1 eval] wrote:', out_dir)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
