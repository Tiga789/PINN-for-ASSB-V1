from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_nn.data import build_dataset
from gv1.p2dlite_rg_nn.metrics import compute_rg_metrics, thresholds_status
from gv1.p2dlite_rg_nn.model import build_model
from gv1.p2dlite_rg_nn.train_eval import standardize_x, standardize_y, unstandardize_y
from gv1.p2dlite_rg_nn.utils import ensure_clean_or_allowed, load_json, set_seed, write_csv, write_json


def parse_args():
    p = argparse.ArgumentParser(description='D15-P1 train 8-cell P2Dlite-RG closed-set NN smoke.')
    p.add_argument('--softlabel-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--config', default='configs/d15_p1_nn_smoke_config.json')
    p.add_argument('--filename', default='solution_softlabels.npz')
    p.add_argument('--allow-overwrite', action='store_true')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--max-time-points-per-profile', type=int, default=None)
    p.add_argument('--max-val-points-per-profile', type=int, default=None)
    p.add_argument('--quick', action='store_true', help='Very short debug run; not for scorecard promotion.')
    return p.parse_args()


def _device(name: str):
    import torch
    if name == 'auto' or not name:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def main() -> int:
    args = parse_args()
    import torch
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    cfg = load_json(args.config)
    data_cfg = cfg.get('data', {})
    tr_cfg = cfg.get('training', {})
    model_cfg = cfg.get('model', {})
    thresholds = cfg.get('scorecard_thresholds', {})
    seed = int(args.seed if args.seed is not None else data_cfg.get('random_seed', 151))
    set_seed(seed)
    out_dir = ensure_clean_or_allowed(args.out_dir, allow_overwrite=args.allow_overwrite)
    model_dir = out_dir / 'model'
    model_dir.mkdir(parents=True, exist_ok=True)
    max_train = int(args.max_time_points_per_profile if args.max_time_points_per_profile is not None else data_cfg.get('max_time_points_per_profile_train', 8192))
    max_val = int(args.max_val_points_per_profile if args.max_val_points_per_profile is not None else data_cfg.get('max_time_points_per_profile_val', 2048))
    epochs = int(args.epochs if args.epochs is not None else tr_cfg.get('epochs', 1200))
    batch_size = int(args.batch_size if args.batch_size is not None else tr_cfg.get('batch_size', 8192))
    if args.quick:
        epochs = min(epochs, 12)
        max_train = min(max_train, 512)
        max_val = min(max_val, 128)
        batch_size = min(batch_size, 1024)
    print('[D15-P1 train] loading dataset...', flush=True)
    bundle = build_dataset(
        args.softlabel_dir,
        filename=args.filename,
        max_train_per_profile=max_train,
        max_val_per_profile=max_val,
        include_profile_onehot=bool(data_cfg.get('include_profile_onehot', True)),
        seed=seed,
    )
    device = _device(args.device if args.device is not None else tr_cfg.get('device', 'auto'))
    print(f'[D15-P1 train] device={device}; train={bundle.X_train.shape}; val={bundle.X_val.shape}; targets={bundle.Y_train.shape[1]}', flush=True)
    Xtr = standardize_x(bundle.X_train, bundle.x_mean, bundle.x_std)
    Ytr = standardize_y(bundle.Y_train, bundle.y_mean, bundle.y_std)
    Xva = standardize_x(bundle.X_val, bundle.x_mean, bundle.x_std)
    Yva = standardize_y(bundle.Y_val, bundle.y_mean, bundle.y_std)
    Xtr_t_all = torch.from_numpy(Xtr).to(device)
    Ytr_t_all = torch.from_numpy(Ytr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)
    Yva_t = torch.from_numpy(Yva).to(device)
    model = build_model(bundle.X_train.shape[1], bundle.Y_train.shape[1], model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr_cfg.get('learning_rate', 0.0015)), weight_decay=float(tr_cfg.get('weight_decay', 1e-6)))
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=int(tr_cfg.get('lr_decay_every', 50)), gamma=float(tr_cfg.get('lr_decay_gamma', 0.985)))
    loss_fn = torch.nn.MSELoss()
    grad_clip = float(tr_cfg.get('grad_clip_norm', 5.0))
    log_every = int(tr_cfg.get('log_every', 25))
    patience = int(tr_cfg.get('early_stop_patience', 250))
    min_delta = float(tr_cfg.get('early_stop_min_delta', 1e-7))
    history = []
    best_val = float('inf')
    best_epoch = -1
    best_path = model_dir / 'best.pt'
    best_state_dict_cpu = None
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        # Manual mini-batching is intentionally used instead of DataLoader to avoid
        # Windows/PowerShell worker edge cases and to keep GTX 1080 Ti runs simple.
        perm = torch.randperm(Xtr_t_all.shape[0], device=device)
        for start_i in range(0, Xtr_t_all.shape[0], batch_size):
            idx = perm[start_i:start_i + batch_size]
            xb = Xtr_t_all.index_select(0, idx)
            yb = Ytr_t_all.index_select(0, idx)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(Xva_t)
            val_loss = float(loss_fn(val_pred, Yva_t).detach().cpu())
        tr_loss = float(np.mean(train_losses)) if train_losses else float('nan')
        lr = float(opt.param_groups[0]['lr'])
        row = {'epoch': epoch, 'train_mse_std': tr_loss, 'val_mse_std': val_loss, 'lr': lr, 'elapsed_s': time.time() - start}
        history.append(row)
        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_epoch = epoch
            best_state_dict_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({'model_state_dict': best_state_dict_cpu}, best_path)
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(f'[D15-P1 train] epoch={epoch:05d} train={tr_loss:.6e} val={val_loss:.6e} best={best_val:.6e}@{best_epoch}', flush=True)
        if epoch - best_epoch > patience:
            print(f'[D15-P1 train] early stop at epoch {epoch}; best_epoch={best_epoch}', flush=True)
            break
    # Restore best without re-loading from disk to avoid platform-specific torch.load stalls.
    if best_state_dict_cpu is None:
        best_state_dict_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict({k: v.to(device) for k, v in best_state_dict_cpu.items()})
    model.eval()
    with torch.no_grad():
        val_pred_std = model(Xva_t).detach().cpu().numpy()
    val_pred = unstandardize_y(val_pred_std, bundle.y_mean, bundle.y_std)
    val_metrics = compute_rg_metrics(bundle.Y_val, val_pred, bundle.target_slices)
    status = thresholds_status(dict(val_metrics), thresholds)
    train_summary = {
        'stage': 'D15-P1 8-cell P2Dlite-RG closed-set NN smoke',
        'softlabel_dir': str(args.softlabel_dir),
        'out_dir': str(out_dir),
        'config': str(args.config),
        'quick_debug_run': bool(args.quick),
        'device': str(device),
        'seed': seed,
        'epochs_requested': epochs,
        'epochs_completed': int(history[-1]['epoch']) if history else 0,
        'best_epoch': int(best_epoch),
        'best_val_mse_std': float(best_val),
        'profile_count': int(len(bundle.profile_ids)),
        'profile_ids': bundle.profile_ids,
        'profile_paths': bundle.profile_paths,
        'nr_a': int(bundle.nr_a),
        'nr_c': int(bundle.nr_c),
        'feature_dim': int(bundle.X_train.shape[1]),
        'target_dim': int(bundle.Y_train.shape[1]),
        'train_points': int(bundle.X_train.shape[0]),
        'val_points': int(bundle.X_val.shape[0]),
        'val_metrics_sampled': val_metrics,
        'val_scorecard_sampled': status,
        'overall_status_sampled_val': 'PASS' if status.get('overall_status') == 'PASS' and not args.quick else ('DEBUG_ONLY' if args.quick else 'REVIEW'),
        'notes': cfg.get('audit_notes', []),
    }
    state = {
        'model_config': model_cfg,
        'training_config': tr_cfg,
        'data_config': data_cfg,
        'feature_names': bundle.feature_names,
        'target_names': bundle.target_names,
        'target_slices': bundle.target_slices,
        'profile_ids': bundle.profile_ids,
        'profile_paths': bundle.profile_paths,
        'x_mean': bundle.x_mean,
        'x_std': bundle.x_std,
        'y_mean': bundle.y_mean,
        'y_std': bundle.y_std,
        'nr_a': bundle.nr_a,
        'nr_c': bundle.nr_c,
        'include_profile_onehot': bool(data_cfg.get('include_profile_onehot', True)),
        'input_dim': int(bundle.X_train.shape[1]),
        'output_dim': int(bundle.Y_train.shape[1]),
        'seed': seed,
    }
    torch.save({'model_state_dict': best_state_dict_cpu, 'state': state}, model_dir / 'best_with_state.pt')
    # Also write numpy stats for transparent non-torch inspection.
    np.savez_compressed(
        model_dir / 'normalization_and_schema.npz',
        x_mean=bundle.x_mean,
        x_std=bundle.x_std,
        y_mean=bundle.y_mean,
        y_std=bundle.y_std,
        feature_names=np.array(bundle.feature_names),
        target_names=np.array(bundle.target_names),
        profile_ids=np.array(bundle.profile_ids),
    )
    write_json(train_summary, out_dir / 'D15_P1_TRAINING_SUMMARY.json')
    write_csv(history, out_dir / 'training_history.csv')
    write_json(bundle.train_meta, out_dir / 'D15_P1_DATASET_SAMPLING_SUMMARY.json')
    print('[D15-P1 train] wrote:', out_dir)
    print('[D15-P1 train] sampled val status:', train_summary['overall_status_sampled_val'])
    return 0 if not args.quick else 0

if __name__ == '__main__':
    raise SystemExit(main())
