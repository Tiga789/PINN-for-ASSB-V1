
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_nn.data import (
    build_features,
    build_targets,
    discover_npz,
    load_profile_arrays,
    profile_id_from_path,
)
from gv1.p2dlite_rg_nn.metrics import compute_rg_metrics, thresholds_status
from gv1.p2dlite_rg_nn.model import build_model
from gv1.p2dlite_rg_nn.train_eval import predict_numpy
from gv1.p2dlite_rg_nn.utils import load_json, write_csv, write_json
from gv1.p2dlite_rg_nn_precision.audit import (
    aggregate_rows,
    audit_prediction_file,
    precision_status,
)
try:
    from gv1.p2dlite_rg_boundary.projection import apply_theta_projection, theta_outside_counts
except Exception:  # pragma: no cover - project-version fallback
    apply_theta_projection = None
    theta_outside_counts = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            'D16-P5A fixed existing-model transfer evaluation. '
            'This script uses D15 P2/P1/P3 best_with_state.pt checkpoints only; '
            'it never treats D14/D12 best.pt files as compatible D15-RG checkpoints.'
        )
    )
    p.add_argument('--softlabel-dir', required=True, help='D15 ALL55 final soft-label root.')
    p.add_argument('--run-dir', required=True, help='D16-P5A output root.')
    p.add_argument('--model-dir', default='auto', help='D15 existing model run dir. Use auto to find best_with_state.pt.')
    p.add_argument('--cache-root', default=r'E:\XJTU battery dataset\_gv1_cache')
    p.add_argument('--config', default='configs/d15_p2_precision_benchmark_config.json')
    p.add_argument('--filename', default='solution_softlabels.npz')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=65536)
    p.add_argument('--eval-stride', type=int, default=None)
    p.add_argument('--limit-cells', type=int, default=None, help='Debug: evaluate first N profiles only.')
    p.add_argument('--allow-overwrite', action='store_true')
    p.add_argument('--primary-mode', choices=['raw', 'projected'], default='projected', help='Which prediction is exposed to precision_audit in eval_full_profiles/predictions.')
    p.add_argument('--theta-min', type=float, default=1e-4)
    p.add_argument('--theta-max', type=float, default=0.9999)
    p.add_argument('--route-unseen', choices=['auto', 'first', 'strict'], default='auto', help='How ALL55 unseen profiles are mapped into the existing model one-hot space.')
    p.add_argument('--no-audit', action='store_true', help='Only generate predictions/metrics; skip D15-P2 precision audit.')
    return p.parse_args()


def _device(name: str):
    import torch
    if name == 'auto' or not name:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def _json_default(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if not math.isfinite(v) else v
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return str(x)


def _dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _clean_dir(path: Path, allow: bool) -> Path:
    if path.exists() and any(path.iterdir()):
        if not allow:
            raise FileExistsError(f'Output directory exists and is non-empty: {path}; pass --allow-overwrite')
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_file(model_dir: Path) -> Path:
    # D15-P1/P2/P3 train scripts save model/best_with_state.pt.
    for p in [model_dir / 'model' / 'best_with_state.pt', model_dir / 'best_with_state.pt']:
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError(f'Could not find D15-compatible best_with_state.pt under {model_dir} or {model_dir / "model"}')


def _score_checkpoint_path(p: Path) -> int:
    s = str(p).lower().replace('\\', '/')
    score = 0
    if 'd15' in s: score += 100
    if 'p2' in s: score += 80
    if 'p3c' in s: score += 70
    if 'p3b' in s: score += 65
    if 'batch2' in s: score += 55
    if 'p1' in s: score += 50
    if 'rg' in s: score += 35
    if 'precision' in s: score += 25
    if 'closedset' in s: score += 15
    if 'd14' in s: score -= 120
    if 'd12' in s or 'd11' in s or 'd10' in s or 'batch134' in s: score -= 80
    return score


def discover_model_dir(requested: str, cache_root: Path, project_root: Path, run_dir: Path) -> Tuple[Path, Path, Dict[str, Any]]:
    discovery: Dict[str, Any] = {
        'requested_model_dir': requested,
        'cache_root': str(cache_root),
        'project_root': str(project_root),
        'policy': 'Only best_with_state.pt is accepted. D14/D12 best.pt checkpoints are intentionally ignored.',
        'candidates': [],
    }
    if requested and requested.lower() != 'auto':
        md = Path(requested)
        ck = checkpoint_file(md)
        discovery['selected'] = {'model_dir': str(md), 'checkpoint': str(ck), 'reason': 'explicit_model_dir'}
        return md, ck, discovery

    canonical_dirs = [
        cache_root / 'xjtu_d15_p2_rg_precision_benchmark',
        cache_root / 'xjtu_d15_p1_rg_closedset_nn_smoke',
    ]
    # Also allow common suffix variants if users kept multiple D15 runs.
    exact_candidates: List[Path] = []
    for md in canonical_dirs:
        try:
            ck = checkpoint_file(md)
            exact_candidates.append(ck)
        except Exception:
            pass

    search_roots = []
    for r in [cache_root, project_root]:
        if r.exists() and r.is_dir():
            search_roots.append(r)
    found: Dict[str, Path] = {str(p): p for p in exact_candidates}
    for root in search_roots:
        try:
            # Limit to best_with_state.pt only. Do not scan best.pt.
            for p in root.rglob('best_with_state.pt'):
                found[str(p)] = p
        except Exception as exc:
            discovery.setdefault('search_errors', []).append(f'{root}: {exc!r}')

    ranked = sorted(found.values(), key=lambda p: _score_checkpoint_path(p), reverse=True)
    for rank, ck in enumerate(ranked[:30], 1):
        md = ck.parent.parent if ck.parent.name == 'model' else ck.parent
        discovery['candidates'].append({
            'rank': rank,
            'score': _score_checkpoint_path(ck),
            'model_dir': str(md),
            'checkpoint': str(ck),
        })
    if not ranked:
        _dump_json(discovery, run_dir / 'D16_P5A_MODEL_DISCOVERY_FAILURE.json')
        raise FileNotFoundError(
            'No D15-compatible best_with_state.pt found. Expected one of:\n'
            f'  {cache_root / "xjtu_d15_p2_rg_precision_benchmark" / "model" / "best_with_state.pt"}\n'
            f'  {cache_root / "xjtu_d15_p1_rg_closedset_nn_smoke" / "model" / "best_with_state.pt"}\n'
            'This script intentionally ignores D14/D12 best.pt files because they do not contain the D15-RG state schema.'
        )
    ck = ranked[0]
    md = ck.parent.parent if ck.parent.name == 'model' else ck.parent
    discovery['selected'] = {'model_dir': str(md), 'checkpoint': str(ck), 'reason': 'best_scored_best_with_state_candidate'}
    return md, ck, discovery


def _scalar(x: Any, default: str = '') -> str:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return default
        return str(arr.reshape(-1)[0])
    except Exception:
        return default


def _profile_metadata(npz_path: Path) -> Dict[str, str]:
    out = {'batch': '', 'protocol': '', 'cell_uid': ''}
    try:
        with np.load(npz_path, allow_pickle=True) as z:
            for k in out:
                if k in z.files:
                    out[k] = _scalar(z[k], '')
    except Exception:
        pass
    return out


def _normalize(s: str) -> str:
    return str(s or '').lower().replace('\\', '/').replace('_', '-').replace(' ', '')


def build_seen_meta(profile_ids: Sequence[str], model_dir: Path) -> List[Dict[str, Any]]:
    seen = []
    for i, pid in enumerate(profile_ids):
        seen.append({
            'seen_index': i,
            'seen_profile_id': str(pid),
            'norm': _normalize(pid),
        })
    return seen


def route_profile(pid: str, meta: Mapping[str, str], seen: Sequence[Mapping[str, Any]], mode: str) -> Tuple[int, str, int]:
    if not seen:
        return 0, 'no_seen_profiles_fallback_0', 0
    norm_pid = _normalize(pid)
    for s in seen:
        if norm_pid == s['norm']:
            return int(s['seen_index']), 'exact_profile_id', 1000
    if mode == 'strict':
        raise KeyError(f'Profile {pid!r} is not in existing model profile_ids and --route-unseen=strict')
    if mode == 'first':
        return int(seen[0]['seen_index']), 'first_seen_profile', 0

    batch = _normalize(meta.get('batch', ''))
    protocol = _normalize(meta.get('protocol', ''))
    cell = _normalize(meta.get('cell_uid', ''))
    # Also mine tokens from path if metadata is missing.
    combined = '|'.join([norm_pid, batch, protocol, cell])

    best_idx = int(seen[0]['seen_index'])
    best_score = -10**9
    best_reason = 'auto_fallback_first'
    for s in seen:
        sn = str(s['norm'])
        score = 0
        reasons = []
        # Protocol/family matches.
        for tok in ['r2.5', 'r25', 'r3', 'geo', 'random', 'randomwalk', '2c', '3c', 'batch-2', 'batch2']:
            if tok in combined and tok in sn:
                score += 100
                reasons.append(f'token:{tok}')
        # Batch matches.
        for b in ['batch-1', 'batch-2', 'batch-3', 'batch-4', 'batch-5', 'batch-6', 'batch1', 'batch2', 'batch3', 'batch4', 'batch5', 'batch6']:
            if b in combined and b in sn:
                score += 60
                reasons.append(f'batch:{b}')
        # Cell UID partial match.
        if cell and cell in sn:
            score += 30
            reasons.append('cell_uid')
        # Batch-2 3C full-cycle is closer to 2C fixed full-cycle if no Batch-2 seen.
        if ('batch-2' in combined or 'batch2' in combined or '3c' in combined) and ('2c' in sn or 'batch-1' in sn):
            score += 40
            reasons.append('batch2_to_2c_proxy')
        # If random/GEO absent in seen, fall back to any active protocol with nearest batch token.
        if score == 0 and ('batch-5' in combined or 'batch5' in combined) and ('random' in sn or 'batch-5' in sn):
            score += 50
            reasons.append('batch5_proxy')
        if score == 0 and ('batch-6' in combined or 'batch6' in combined) and ('geo' in sn or 'batch-6' in sn):
            score += 50
            reasons.append('batch6_proxy')
        if score > best_score:
            best_score = score
            best_idx = int(s['seen_index'])
            best_reason = '+'.join(reasons) if reasons else 'auto_fallback_first'
    return best_idx, best_reason, int(best_score)


def _status_from_scorecard(score: Mapping[str, Any]) -> str:
    return str(score.get('overall_status', 'FAIL'))


def aggregate_group(rows: List[Mapping[str, Any]], keys: Sequence[str], out_path: Path) -> None:
    groups: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    for r in rows:
        key = tuple(str(r.get(k, '')) for k in keys)
        groups.setdefault(key, []).append(r)
    agg_rows: List[Dict[str, Any]] = []
    metric_keys = sorted({
        k for r in rows for k, v in r.items()
        if isinstance(v, (int, float, np.integer, np.floating)) and any(s in k for s in ['mae', 'rmse', 'corr', 'r2', 'outside', 'max_abs'])
    })
    for key, rs in groups.items():
        out = {k: key[i] for i, k in enumerate(keys)}
        out['profile_count'] = len(rs)
        for mk in metric_keys:
            vals = []
            for r in rs:
                try:
                    v = float(r.get(mk, float('nan')))
                    if math.isfinite(v):
                        vals.append(v)
                except Exception:
                    pass
            if vals:
                out[f'{mk}_mean'] = float(np.mean(vals))
                out[f'{mk}_max'] = float(np.max(vals))
                out[f'{mk}_min'] = float(np.min(vals))
        agg_rows.append(out)
    write_csv(agg_rows, out_path)


def main() -> int:
    args = parse_args()
    t0 = time.time()
    run_dir = Path(args.run_dir)
    if run_dir.exists() and args.allow_overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = run_dir / 'eval_full_profiles'
    pred_primary = eval_dir / 'predictions'
    pred_raw = eval_dir / 'predictions_raw'
    pred_proj = eval_dir / 'predictions_projected'
    audit_dir = run_dir / 'precision_audit'
    for d in [eval_dir, pred_primary, pred_raw, pred_proj, audit_dir]:
        d.mkdir(parents=True, exist_ok=True)

    cache_root = Path(args.cache_root)
    soft_root = Path(args.softlabel_dir)
    if not soft_root.exists():
        raise FileNotFoundError(f'Softlabel root not found: {soft_root}')
    cfg = load_json(args.config)
    thresholds = cfg.get('scorecard_thresholds', {})
    eval_stride = int(args.eval_stride if args.eval_stride is not None else cfg.get('data', {}).get('eval_stride', 1))

    model_dir, model_file, discovery = discover_model_dir(args.model_dir, cache_root, ROOT, run_dir)
    _dump_json(discovery, run_dir / 'D16_P5A_MODEL_DISCOVERY.json')

    import torch
    device = _device(args.device)
    ck = torch.load(model_file, map_location=device, weights_only=False)
    if 'state' not in ck:
        raise KeyError(f'Checkpoint is not a D15-RG best_with_state.pt file: {model_file}; missing top-level state')
    state = ck['state']
    profile_ids = list(state.get('profile_ids', []))
    profile_count = len(profile_ids)
    model = build_model(int(state['input_dim']), int(state['output_dim']), state['model_config']).to(device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    seen_meta = build_seen_meta(profile_ids, model_dir)

    files = discover_npz(soft_root, filename=args.filename)
    if args.limit_cells is not None and args.limit_cells > 0:
        files = files[:int(args.limit_cells)]
    if not files:
        raise FileNotFoundError(f'No {args.filename} found under {soft_root}')

    rows: List[Dict[str, Any]] = []
    routing_rows: List[Dict[str, Any]] = []
    all_true: List[np.ndarray] = []
    all_raw: List[np.ndarray] = []
    all_projected: List[np.ndarray] = []
    failures: List[str] = []
    print(f'[D16-P5A v4] model_dir={model_dir}', flush=True)
    print(f'[D16-P5A v4] model_file={model_file}', flush=True)
    print(f'[D16-P5A v4] softlabel_profiles={len(files)}; model_profile_count={profile_count}; device={device}; primary_mode={args.primary_mode}', flush=True)

    for default_i, npz_path in enumerate(files, 1):
        pid = profile_id_from_path(npz_path, soft_root)
        meta = _profile_metadata(npz_path)
        try:
            route_idx, route_reason, route_score = route_profile(pid, meta, seen_meta, args.route_unseen)
            prof = load_profile_arrays(npz_path, soft_root)
            X, _ = build_features(prof, route_idx, profile_count, include_profile_onehot=bool(state.get('include_profile_onehot', True)))
            Y, _, _ = build_targets(prof)
            stride = max(1, int(eval_stride))
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
                batch_size=int(args.batch_size),
            )
            if apply_theta_projection is not None:
                Y_projected = apply_theta_projection(Y_raw, state['target_slices'], theta_min=args.theta_min, theta_max=args.theta_max)
            else:
                Y_projected = np.asarray(Y_raw, dtype=np.float32).copy()
                for key in ['theta_a', 'theta_c']:
                    s, e = state['target_slices'][key]
                    Y_projected[:, s:e] = np.clip(Y_projected[:, s:e], args.theta_min, args.theta_max)

            raw_metrics = compute_rg_metrics(Y_eval, Y_raw, state['target_slices'])
            projected_metrics = compute_rg_metrics(Y_eval, Y_projected, state['target_slices'])
            raw_score = thresholds_status(dict(raw_metrics), thresholds)
            projected_score = thresholds_status(dict(projected_metrics), thresholds)
            row: Dict[str, Any] = {
                'profile_id': pid,
                'npz_path': str(npz_path),
                'batch': meta.get('batch', ''),
                'protocol': meta.get('protocol', ''),
                'cell_uid': meta.get('cell_uid', ''),
                'n_eval': int(Y_eval.shape[0]),
                'eval_stride': int(stride),
                'route_seen_index': int(route_idx),
                'route_seen_profile_id': profile_ids[route_idx] if 0 <= route_idx < len(profile_ids) else '',
                'route_reason': route_reason,
                'route_score': int(route_score),
                'raw_status': raw_score.get('overall_status', 'FAIL'),
                'projected_status': projected_score.get('overall_status', 'FAIL'),
            }
            for k, v in raw_metrics.items(): row[f'raw_{k}'] = v
            for k, v in projected_metrics.items(): row[f'projected_{k}'] = v
            rows.append(row)
            routing_rows.append({k: row[k] for k in ['profile_id','batch','protocol','cell_uid','route_seen_index','route_seen_profile_id','route_reason','route_score']})

            safe = pid.replace('/', '__').replace('\\', '__').replace(':', '_')
            common = dict(
                t_global_s=t_eval.astype(np.float32),
                y_true=Y_eval.astype(np.float32),
                target_names=np.array(state['target_names']),
                feature_names=np.array(state['feature_names']),
                profile_id=np.array(pid),
                routed_seen_profile_id=np.array(str(profile_ids[route_idx] if 0 <= route_idx < len(profile_ids) else '')),
                route_reason=np.array(str(route_reason)),
            )
            np.savez_compressed(pred_raw / f'{safe}_raw_prediction.npz', **common, y_pred=Y_raw.astype(np.float32), prediction_mode=np.array('raw'))
            np.savez_compressed(pred_proj / f'{safe}_projected_prediction.npz', **common, y_pred=Y_projected.astype(np.float32), prediction_mode=np.array('projected'))
            primary_pred = Y_projected if args.primary_mode == 'projected' else Y_raw
            np.savez_compressed(pred_primary / f'{safe}_prediction.npz', **common, y_pred=primary_pred.astype(np.float32), prediction_mode=np.array(args.primary_mode))

            all_true.append(Y_eval)
            all_raw.append(Y_raw)
            all_projected.append(Y_projected)
            print(f'[D16-P5A v4] {default_i}/{len(files)} {pid}: route={route_idx}({route_reason}); raw_phis={raw_metrics.get("phis_c_mae"):.6g}; proj_theta_out={projected_metrics.get("pred_theta_outside_fraction"):.6g}', flush=True)
        except Exception as exc:
            failures.append(f'{pid}: {exc!r}')
            print(f'[D16-P5A v4] FAIL {pid}: {exc!r}', flush=True)

    if not rows:
        raise RuntimeError('No profiles were evaluated; see failures in console output.')

    YT = np.concatenate(all_true, axis=0)
    YR = np.concatenate(all_raw, axis=0)
    YP = np.concatenate(all_projected, axis=0)
    raw_global = compute_rg_metrics(YT, YR, state['target_slices'])
    projected_global = compute_rg_metrics(YT, YP, state['target_slices'])
    raw_status = thresholds_status(dict(raw_global), thresholds)
    projected_status = thresholds_status(dict(projected_global), thresholds)

    write_csv(rows, eval_dir / 'D16_P5A_METRICS_BY_PROFILE.csv')
    write_json(rows, eval_dir / 'D16_P5A_METRICS_BY_PROFILE.json')
    write_csv(routing_rows, eval_dir / 'D16_P5A_ROUTING_TABLE.csv')
    write_json(routing_rows, eval_dir / 'D16_P5A_ROUTING_TABLE.json')
    aggregate_group(rows, ['batch'], eval_dir / 'D16_P5A_BATCH_METRICS.csv')
    aggregate_group(rows, ['protocol'], eval_dir / 'D16_P5A_PROTOCOL_METRICS.csv')
    if failures:
        write_csv([{'failure': f} for f in failures], eval_dir / 'D16_P5A_FAILURES.csv')

    eval_summary = {
        'stage': 'D16-P5A existing D15-RG model transfer evaluation on ALL55',
        'softlabel_dir': str(soft_root),
        'run_dir': str(run_dir),
        'eval_dir': str(eval_dir),
        'model_dir': str(model_dir),
        'model_file': str(model_file),
        'profile_count_discovered': len(files),
        'profile_count_evaluated': len(rows),
        'prediction_file_count_primary': len(list(pred_primary.glob('*.npz'))),
        'primary_mode_for_precision_audit': args.primary_mode,
        'model_seen_profile_count': int(profile_count),
        'model_seen_profile_ids': profile_ids,
        'eval_stride': int(eval_stride),
        'batch_size': int(args.batch_size),
        'device': str(device),
        'raw_global_metrics': raw_global,
        'projected_global_metrics': projected_global,
        'raw_scorecard': raw_status,
        'projected_scorecard': projected_status,
        'failures': failures,
        'overall_status': 'FAIL' if failures else ('PASS' if projected_status.get('overall_status') == 'PASS' else 'REVIEW'),
        'elapsed_s': float(time.time() - t0),
        'interpretation': 'D16-P5A tests transfer of an already trained D15-RG NN to ALL55 labels; it is not a new ALL55 unified training result.',
    }
    write_json(eval_summary, eval_dir / 'D16_P5A_EVAL_SUMMARY.json')
    # D15-P2 alias is intentionally written so existing scorecard collectors do not see MISSING eval status.
    d15_alias = dict(eval_summary)
    d15_alias['stage_alias'] = 'D15-P2 style eval summary generated by D16-P5A transfer evaluator'
    d15_alias['global_metrics'] = projected_global if args.primary_mode == 'projected' else raw_global
    d15_alias['scorecard'] = projected_status if args.primary_mode == 'projected' else raw_status
    d15_alias['overall_status'] = eval_summary['overall_status']
    write_json(d15_alias, eval_dir / 'D15_P2_EVAL_SUMMARY.json')

    audit_summary: Dict[str, Any] = {'overall_status': 'SKIPPED', 'reason': '--no-audit'}
    if not args.no_audit:
        cfg_audit = cfg.get('precision_audit', {})
        audit_rows: List[Dict[str, Any]] = []
        top_rows: List[Dict[str, Any]] = []
        cycle_rows: List[Dict[str, Any]] = []
        audit_failures: List[str] = []
        preds = sorted(pred_primary.glob('*.npz'))
        print(f'[D16-P5A v4] precision audit scanning primary predictions: {len(preds)}', flush=True)
        for p in preds:
            try:
                r, top, cyc = audit_prediction_file(p, soft_root, cfg_audit, filename=args.filename)
                audit_rows.append(r)
                top_rows.extend(top)
                cycle_rows.extend(cyc)
            except Exception as exc:
                audit_failures.append(f'{p}: {exc!r}')
                print(f'[D16-P5A v4] AUDIT FAIL {p}: {exc!r}', flush=True)
        aggregate = aggregate_rows(audit_rows)
        status = precision_status(audit_rows, aggregate, cfg_audit)
        if audit_failures:
            status['overall_status'] = 'FAIL'
            status['read_failures'] = audit_failures
        audit_summary = {
            'stage': 'D16-P5A precision audit using D15-P2 audit logic',
            'softlabel_dir': str(soft_root),
            'eval_dir': str(eval_dir),
            'prediction_root': str(pred_primary),
            'prediction_file_count': len(preds),
            'profile_count_audited': len(audit_rows),
            'primary_mode': args.primary_mode,
            'aggregate': aggregate,
            'precision_status': status,
            'overall_status': status.get('overall_status', 'FAIL'),
            'failures': audit_failures,
            'notes': cfg.get('audit_notes', []),
        }
        write_csv(audit_rows, audit_dir / 'D15_P2_PRECISION_AUDIT_BY_PROFILE.csv')
        write_json(audit_rows, audit_dir / 'D15_P2_PRECISION_AUDIT_BY_PROFILE.json')
        write_csv(top_rows, audit_dir / 'D15_P2_TOPK_ERROR_WINDOWS.csv')
        write_json(top_rows, audit_dir / 'D15_P2_TOPK_ERROR_WINDOWS.json')
        write_csv(cycle_rows, audit_dir / 'D15_P2_CYCLE_LEVEL_AUDIT.csv')
        write_json(audit_summary, audit_dir / 'D15_P2_PRECISION_AUDIT_SUMMARY.json')

    operational_status = 'PASS' if len(rows) == len(files) and len(list(pred_primary.glob('*.npz'))) == len(files) and not failures else 'FAIL'
    if operational_status != 'PASS':
        final_status = 'FAIL'
    elif audit_summary.get('overall_status') == 'PASS' and eval_summary['overall_status'] == 'PASS':
        final_status = 'PASS'
    elif audit_summary.get('overall_status') in {'PASS', 'REVIEW', 'SKIPPED'} and eval_summary['overall_status'] in {'PASS', 'REVIEW'}:
        final_status = 'REVIEW'
    else:
        final_status = 'FAIL'
    final = {
        'stage': 'D16-P5A final scorecard v4',
        'final_status': final_status,
        'operational_status': operational_status,
        'run_dir': str(run_dir),
        'softlabel_dir': str(soft_root),
        'model_dir': str(model_dir),
        'model_file': str(model_file),
        'eval_dir': str(eval_dir),
        'audit_dir': str(audit_dir),
        'profile_count_discovered': len(files),
        'profile_count_evaluated': len(rows),
        'prediction_file_count_primary': len(list(pred_primary.glob('*.npz'))),
        'primary_mode': args.primary_mode,
        'raw_eval_status': raw_status.get('overall_status'),
        'projected_eval_status': projected_status.get('overall_status'),
        'precision_audit_status': audit_summary.get('overall_status'),
        'raw_key_metrics': raw_global,
        'projected_key_metrics': projected_global,
        'precision_audit_summary': audit_summary.get('aggregate', {}),
        'failures': failures + audit_summary.get('failures', []),
        'routing_table': str(eval_dir / 'D16_P5A_ROUTING_TABLE.csv'),
        'metrics_by_profile': str(eval_dir / 'D16_P5A_METRICS_BY_PROFILE.csv'),
        'batch_metrics': str(eval_dir / 'D16_P5A_BATCH_METRICS.csv'),
        'protocol_metrics': str(eval_dir / 'D16_P5A_PROTOCOL_METRICS.csv'),
        'prediction_root_primary': str(pred_primary),
        'elapsed_s': float(time.time() - t0),
    }
    write_json(final, run_dir / 'D16_P5A_FINAL_SCORECARD.json')
    print('[D16-P5A v4] operational_status:', operational_status, flush=True)
    print('[D16-P5A v4] final_status:', final_status, flush=True)
    print('[D16-P5A v4] primary predictions:', pred_primary, flush=True)
    print('[D16-P5A v4] final scorecard:', run_dir / 'D16_P5A_FINAL_SCORECARD.json', flush=True)
    return 0 if operational_status == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
