from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_boundary.projection import (
    apply_theta_projection,
    compare_mae_nonregression,
    theta_outside_counts,
    top_theta_outside_points,
)
from gv1.p2dlite_rg_nn.data import build_features, build_targets, discover_npz, load_profile_arrays, profile_id_from_path
from gv1.p2dlite_rg_nn.metrics import compute_rg_metrics, thresholds_status
from gv1.p2dlite_rg_nn.model import build_model
from gv1.p2dlite_rg_nn.train_eval import predict_numpy
from gv1.p2dlite_rg_nn.utils import ensure_clean_or_allowed, load_json, write_csv, write_json


def parse_args():
    p = argparse.ArgumentParser(description='D15-P3B evaluate raw vs theta-projected Batch-2 NN predictions on full P2Dlite-RG profiles.')
    p.add_argument('--softlabel-dir', required=True)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--config', default='configs/d15_p3b_boundary_repair_config.json')
    p.add_argument('--filename', default='solution_softlabels.npz')
    p.add_argument('--device', default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--eval-stride', type=int, default=None)
    p.add_argument('--allow-overwrite', action='store_true')
    p.add_argument('--save-prediction-npz', action='store_true')
    p.add_argument('--top-k', type=int, default=200)
    return p.parse_args()


def _device(name: str | None):
    import torch
    if name is None or name == 'auto' or not name:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def _model_file(model_dir: Path) -> Path:
    if (model_dir / 'model' / 'best_with_state.pt').exists():
        return model_dir / 'model' / 'best_with_state.pt'
    if (model_dir / 'best_with_state.pt').exists():
        return model_dir / 'best_with_state.pt'
    raise FileNotFoundError(f'Could not find best_with_state.pt under {model_dir} or {model_dir / "model"}')


def _prefix_dict(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f'{prefix}{k}': v for k, v in d.items()}


def _safe_cuda_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {'torch_cuda_available': False}
    try:
        import torch
        info['torch_cuda_available'] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info['cuda_device_count'] = int(torch.cuda.device_count())
            info['cuda_device_name_0'] = str(torch.cuda.get_device_name(0))
            props = torch.cuda.get_device_properties(0)
            info['cuda_total_memory_gb_0'] = round(float(props.total_memory) / (1024 ** 3), 3)
    except Exception as exc:
        info['cuda_info_error'] = repr(exc)
    return info


def main() -> int:
    args = parse_args()
    import torch

    cfg = load_json(args.config)
    out_dir = ensure_clean_or_allowed(args.out_dir, allow_overwrite=args.allow_overwrite)
    thresholds = cfg.get('scorecard_thresholds', {})
    proj_cfg = cfg.get('projection', {})
    inf_cfg = cfg.get('inference', {})
    device = _device(args.device if args.device is not None else inf_cfg.get('device', 'auto'))
    batch_size = int(args.batch_size if args.batch_size is not None else inf_cfg.get('batch_size', 262144))
    eval_stride = int(args.eval_stride if args.eval_stride is not None else cfg.get('data', {}).get('eval_stride', 1))

    root = Path(args.softlabel_dir)
    mf = _model_file(Path(args.model_dir))
    ck = torch.load(mf, map_location=device, weights_only=False)
    state = ck['state']
    model = build_model(int(state['input_dim']), int(state['output_dim']), state['model_config']).to(device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    files = discover_npz(root, filename=args.filename)
    if not files:
        raise FileNotFoundError(f'No soft-label npz files found under {root}')
    profile_ids = list(state.get('profile_ids', []))
    profile_count = len(profile_ids) if profile_ids else len(files)
    if len(files) != profile_count:
        print(f'[D15-P3B] WARNING: found {len(files)} npz files but model profile_count={profile_count}', flush=True)

    rows: List[Dict[str, Any]] = []
    top_rows: List[Dict[str, Any]] = []
    top_proj_error_rows: List[Dict[str, Any]] = []
    all_true: List[np.ndarray] = []
    all_raw: List[np.ndarray] = []
    all_proj: List[np.ndarray] = []

    print(f'[D15-P3B] device={device}; batch_size={batch_size}; eval_stride={eval_stride}', flush=True)
    for default_i, npz_path in enumerate(files):
        pid = profile_id_from_path(npz_path, root)
        try:
            profile_index = profile_ids.index(pid)
        except ValueError:
            profile_index = default_i
        prof = load_profile_arrays(npz_path, root)
        X, _ = build_features(prof, profile_index, profile_count, include_profile_onehot=bool(state.get('include_profile_onehot', True)))
        Y, _, _ = build_targets(prof)
        stride = max(1, eval_stride)
        X_eval = X[::stride]
        Y_eval = Y[::stride]
        t_eval = prof['t'][::stride]
        Y_raw = predict_numpy(
            model,
            X_eval,
            np.asarray(state['x_mean'], dtype=np.float32),
            np.asarray(state['x_std'], dtype=np.float32),
            np.asarray(state['y_mean'], dtype=np.float32),
            np.asarray(state['y_std'], dtype=np.float32),
            device,
            batch_size=batch_size,
        )
        Y_proj = apply_theta_projection(
            Y_raw,
            state['target_slices'],
            theta_min=float(proj_cfg.get('theta_min', 1e-4)),
            theta_max=float(proj_cfg.get('theta_max', 0.9999)),
            apply_to=tuple(proj_cfg.get('apply_to', ['theta_a', 'theta_c'])),
        )
        raw_metrics = compute_rg_metrics(Y_eval, Y_raw, state['target_slices'])
        proj_metrics = compute_rg_metrics(Y_eval, Y_proj, state['target_slices'])
        raw_score = thresholds_status(dict(raw_metrics), thresholds)
        proj_score = thresholds_status(dict(proj_metrics), thresholds)
        nonreg = compare_mae_nonregression(raw_metrics, proj_metrics, thresholds)
        row: Dict[str, Any] = {
            'profile_id': pid,
            'npz_path': str(npz_path),
            'n_eval': int(Y_eval.shape[0]),
            'eval_stride': int(stride),
            'raw_status': raw_score['overall_status'],
            'projected_status': proj_score['overall_status'],
            'nonregression_status': nonreg['overall_status'],
        }
        row.update(_prefix_dict(raw_metrics, 'raw_'))
        row.update(_prefix_dict(proj_metrics, 'projected_'))
        row.update(_prefix_dict(theta_outside_counts(Y_raw, state['target_slices']), 'raw_counts_'))
        row.update(_prefix_dict(theta_outside_counts(Y_proj, state['target_slices']), 'projected_counts_'))
        rows.append(row)
        top_rows.extend(top_theta_outside_points(Y_eval, Y_raw, state['target_slices'], pid, t_eval, top_k=args.top_k))

        # projected top absolute theta errors for post-repair review
        for elec in ('theta_a', 'theta_c'):
            s, e = state['target_slices'][elec]
            err = np.abs(Y_proj[:, s:e] - Y_eval[:, s:e])
            flat_idx = np.argpartition(err.reshape(-1), -min(args.top_k, err.size))[-min(args.top_k, err.size):]
            rr = np.column_stack(np.unravel_index(flat_idx, err.shape))
            for ti, ri in rr:
                top_proj_error_rows.append({
                    'profile_id': pid,
                    'electrode': elec,
                    'time_index': int(ti),
                    'radial_index': int(ri),
                    't_global_s': float(t_eval[ti]) if ti < len(t_eval) else None,
                    'theta_true': float(Y_eval[ti, s + ri]),
                    'theta_pred_projected': float(Y_proj[ti, s + ri]),
                    'theta_pred_raw': float(Y_raw[ti, s + ri]),
                    'abs_error_projected': float(err[ti, ri]),
                })

        if args.save_prediction_npz or bool(inf_cfg.get('save_prediction_npz', False)):
            safe = pid.replace('/', '__').replace('\\', '__').replace(':', '_')
            pred_dir = out_dir / 'predictions'
            pred_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                pred_dir / f'{safe}_raw_projected_prediction.npz',
                t_global_s=t_eval.astype(np.float32),
                y_true=Y_eval.astype(np.float32),
                y_pred_raw=Y_raw.astype(np.float32),
                y_pred_projected=Y_proj.astype(np.float32),
                target_names=np.array(state['target_names']),
                feature_names=np.array(state['feature_names']),
                profile_id=np.array(pid),
            )
        all_true.append(Y_eval)
        all_raw.append(Y_raw)
        all_proj.append(Y_proj)
        print(f'[D15-P3B] {pid}: raw_outside={raw_metrics.get("pred_theta_outside_fraction"):.6g} projected_outside={proj_metrics.get("pred_theta_outside_fraction"):.6g} projected_theta_a_mae={proj_metrics.get("theta_a_mae"):.6g}', flush=True)

    YT = np.concatenate(all_true, axis=0)
    YR = np.concatenate(all_raw, axis=0)
    YP = np.concatenate(all_proj, axis=0)
    raw_global = compute_rg_metrics(YT, YR, state['target_slices'])
    projected_global = compute_rg_metrics(YT, YP, state['target_slices'])
    raw_score = thresholds_status(dict(raw_global), thresholds)
    projected_score = thresholds_status(dict(projected_global), thresholds)
    nonreg_global = compare_mae_nonregression(raw_global, projected_global, thresholds)

    final_status = 'PASS' if projected_score['overall_status'] == 'PASS' and nonreg_global['overall_status'] == 'PASS' else 'REVIEW'
    summary = {
        'stage': 'D15-P3B Batch-2 theta boundary projection repair',
        'softlabel_dir': str(root),
        'model_file': str(mf),
        'out_dir': str(out_dir),
        'profile_count': len(rows),
        'eval_stride': int(eval_stride),
        'batch_size': int(batch_size),
        'device': str(device),
        'gpu_info': _safe_cuda_info(),
        'projection': proj_cfg,
        'raw_global_metrics': raw_global,
        'projected_global_metrics': projected_global,
        'raw_scorecard': raw_score,
        'projected_scorecard': projected_score,
        'nonregression_scorecard': nonreg_global,
        'overall_status': final_status,
        'notes': cfg.get('audit_notes', []),
    }
    top_rows.sort(key=lambda r: float(r.get('outside_distance', 0.0)), reverse=True)
    top_proj_error_rows.sort(key=lambda r: float(r.get('abs_error_projected', 0.0)), reverse=True)

    write_json(summary, out_dir / 'D15_P3B_BOUNDARY_REPAIR_SUMMARY.json')
    write_csv(rows, out_dir / 'D15_P3B_BOUNDARY_REPAIR_BY_PROFILE.csv')
    write_json(rows, out_dir / 'D15_P3B_BOUNDARY_REPAIR_BY_PROFILE.json')
    write_csv(top_rows[:args.top_k], out_dir / 'D15_P3B_TOP_RAW_THETA_OUTSIDE_POINTS.csv')
    write_csv(top_proj_error_rows[:args.top_k], out_dir / 'D15_P3B_TOP_PROJECTED_THETA_ERROR_POINTS.csv')
    print('[D15-P3B] overall_status:', final_status, flush=True)
    print('[D15-P3B] wrote:', out_dir, flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
