from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg.io_utils import (
    copy_profile_metadata,
    discover_softlabel_npz,
    get_cbar_field,
    get_cs_or_theta,
    get_j_field,
    get_time_and_current,
    load_json,
    relative_output_dir,
    save_npz_compressed,
    volume_weights_for_nr,
    weighted_cbar,
    write_json,
)
from gv1.p2dlite_rg.radial_solver import ElectrodeRGParams, generate_rg_profile, infer_surface_flux_from_cbar


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Generate D15-P0 P2Dlite-RG soft labels from source P2Dlite v1 labels.')
    p.add_argument('--source-dir', required=True, help='Source P4B-v3/P2Dlite v1 soft-label directory.')
    p.add_argument('--prior-json', default='configs/P2Dlite_prior_xjtu_lr18650la_rg_v1.json')
    p.add_argument('--output-dir', required=True, help='New output directory, e.g. .../_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_d15p0_8cell')
    p.add_argument('--filename', default='solution_softlabels.npz')
    p.add_argument('--allow-overwrite', action='store_true', help='Allow writing into an existing output directory.')
    p.add_argument('--nr', type=int, default=None, help='Override radial cell count. Default: max(source nr, prior n_r_min).')
    return p.parse_args()


def _electrode_params(prior: Dict[str, Any], electrode: str) -> ElectrodeRGParams:
    rg = prior.get('radial_gradient', {})
    if electrode == 'a':
        spec = prior['electrodes']['negative']
        alpha_D = float(rg.get('alpha_D_negative', 1.0))
        alpha_J = float(rg.get('alpha_J_negative', 1.0))
        name = 'negative_graphite'
    elif electrode == 'c':
        spec = prior['electrodes']['positive']
        alpha_D = float(rg.get('alpha_D_positive', 1.0))
        alpha_J = float(rg.get('alpha_J_positive', 1.0))
        name = 'positive_NCM523'
    else:
        raise ValueError(electrode)
    return ElectrodeRGParams(
        name=name,
        radius_m=float(spec['particle_radius_m']),
        diffusivity_m2_s=float(spec['solid_diffusivity_m2_s']),
        csmax_mol_m3=float(spec['csmax_mol_m3']),
        alpha_D=alpha_D,
        alpha_J=alpha_J,
        gradient_clip_normalized=float(spec.get('gradient_clip_normalized', rg.get('gradient_clip_normalized', 0.12))),
        theta_min_clip=0.0,
        theta_max_clip=1.0,
    )


def _prepare_cbar_and_flux(arrays: Dict[str, Any], electrode: str, t: np.ndarray, cs_source: np.ndarray, params: ElectrodeRGParams, prefer_j: bool = True) -> Tuple[np.ndarray, np.ndarray, str]:
    n_time = len(t)
    weights = volume_weights_for_nr(cs_source.shape[1])
    cbar = get_cbar_field(arrays, electrode, n_time)
    if cbar is None:
        cbar = weighted_cbar(cs_source, weights)
        cbar_source = 'weighted_mean_from_source_cs'
    else:
        cbar_source = 'source_cbar_field'

    J = get_j_field(arrays, electrode, n_time) if prefer_j else None
    if J is not None:
        return cbar.astype(float), J.astype(float), f'source_j_field_from_npz__cbar={cbar_source}'
    J = infer_surface_flux_from_cbar(t, cbar, params.R)
    return cbar.astype(float), J.astype(float), f'inferred_J_from_cbar_derivative__cbar={cbar_source}'


def _process_one(npz_path: Path, source_root: Path, output_root: Path, prior: Dict[str, Any], nr_override: int | None) -> Dict[str, Any]:
    arrays = np.load(npz_path, allow_pickle=True)
    data = {k: arrays[k] for k in arrays.files}
    arrays.close()
    t, I = get_time_and_current(data)
    rg_cfg = prior.get('radial_gradient', {})
    max_sub = float(rg_cfg.get('implicit_step_subdivide_dt_s', 10.0))
    prefer_j = bool(rg_cfg.get('prefer_existing_j_fields', True))
    p_a = _electrode_params(prior, 'a')
    p_c = _electrode_params(prior, 'c')

    cs_a_source, src_key_a = get_cs_or_theta(data, 'a', len(t), p_a.csmax_mol_m3)
    cs_c_source, src_key_c = get_cs_or_theta(data, 'c', len(t), p_c.csmax_mol_m3)
    nr_min = int(rg_cfg.get('n_r_min', 17))
    nr_a = nr_override or max(cs_a_source.shape[1], nr_min)
    nr_c = nr_override or max(cs_c_source.shape[1], nr_min)

    cbar_a, J_a, Jsrc_a = _prepare_cbar_and_flux(data, 'a', t, cs_a_source, p_a, prefer_j=prefer_j)
    cbar_c, J_c, Jsrc_c = _prepare_cbar_and_flux(data, 'c', t, cs_c_source, p_c, prefer_j=prefer_j)

    print(f'  - a: nr={nr_a}, src={src_key_a}, J={Jsrc_a}', flush=True)
    cs_a_rg, diag_a = generate_rg_profile(t, cbar_a, J_a, cs_a_source[0, :], p_a, nr=nr_a, max_substep_s=max_sub)
    print(f'  - c: nr={nr_c}, src={src_key_c}, J={Jsrc_c}', flush=True)
    cs_c_rg, diag_c = generate_rg_profile(t, cbar_c, J_c, cs_c_source[0, :], p_c, nr=nr_c, max_substep_s=max_sub)

    out_arrays = copy_profile_metadata(data)
    # Preserve voltage and phi labels by design. Only radial states and diagnostics are replaced/added.
    out_arrays['cs_a_source_p2dlite_v1'] = cs_a_source.astype(np.float32)
    out_arrays['cs_c_source_p2dlite_v1'] = cs_c_source.astype(np.float32)
    out_arrays['cs_a'] = cs_a_rg.astype(np.float32)
    out_arrays['cs_c'] = cs_c_rg.astype(np.float32)
    out_arrays['theta_a'] = (cs_a_rg / p_a.csmax_mol_m3).astype(np.float32)
    out_arrays['theta_c'] = (cs_c_rg / p_c.csmax_mol_m3).astype(np.float32)
    out_arrays['cbar_a'] = cbar_a.astype(np.float32)
    out_arrays['cbar_c'] = cbar_c.astype(np.float32)
    out_arrays['cs_a_surface'] = cs_a_rg[:, -1].astype(np.float32)
    out_arrays['cs_c_surface'] = cs_c_rg[:, -1].astype(np.float32)
    out_arrays['cs_a_center'] = cs_a_rg[:, 0].astype(np.float32)
    out_arrays['cs_c_center'] = cs_c_rg[:, 0].astype(np.float32)
    out_arrays['grad_a_surface_center'] = diag_a['surface_center'].astype(np.float32)
    out_arrays['grad_c_surface_center'] = diag_c['surface_center'].astype(np.float32)
    out_arrays['grad_a_surface_mean'] = diag_a['surface_mean'].astype(np.float32)
    out_arrays['grad_c_surface_mean'] = diag_c['surface_mean'].astype(np.float32)
    out_arrays['J_a_eff_rg'] = diag_a['J_used'].astype(np.float32)
    out_arrays['J_c_eff_rg'] = diag_c['J_used'].astype(np.float32)
    out_arrays['D_a_eff_rg'] = diag_a['D_eff'].astype(np.float32)
    out_arrays['D_c_eff_rg'] = diag_c['D_eff'].astype(np.float32)
    out_arrays['r_a'] = diag_a['r_norm_centers'].astype(np.float32)
    out_arrays['r_c'] = diag_c['r_norm_centers'].astype(np.float32)
    out_arrays['radial_volume_weights_a'] = diag_a['volume_weights'].astype(np.float32)
    out_arrays['radial_volume_weights_c'] = diag_c['volume_weights'].astype(np.float32)
    out_arrays['radial_solver_version'] = np.array('P2Dlite-RG-v1-implicit-FVM-zero-mean')
    out_arrays['radial_gradient_quality_flag'] = np.array('D15P0_GENERATED_NEEDS_AUDIT')
    out_arrays['source_profile_npz'] = np.array(str(npz_path))
    out_arrays['source_p2dlite_v1_key_a'] = np.array(src_key_a)
    out_arrays['source_p2dlite_v1_key_c'] = np.array(src_key_c)
    out_arrays['source_flux_method_a'] = np.array(Jsrc_a)
    out_arrays['source_flux_method_c'] = np.array(Jsrc_c)
    out_arrays['phis_c_voltage_preserved_from_source'] = np.array(bool(rg_cfg.get('preserve_source_voltage_labels', True)))

    out_dir = relative_output_dir(npz_path, source_root, output_root)
    out_npz = out_dir / 'solution_softlabels.npz'
    save_npz_compressed(out_npz, out_arrays)

    summary = {
        'stage': 'D15-P0 P2Dlite-RG soft-label generation',
        'profile_id': str(out_dir.relative_to(output_root)),
        'source_npz': str(npz_path),
        'output_npz': str(out_npz),
        'time_points': int(len(t)),
        'nr_a': int(nr_a),
        'nr_c': int(nr_c),
        'source_key_a': src_key_a,
        'source_key_c': src_key_c,
        'flux_method_a': Jsrc_a,
        'flux_method_c': Jsrc_c,
        'csmax_a': float(p_a.csmax_mol_m3),
        'csmax_c': float(p_c.csmax_mol_m3),
        'D_eff_a': float(p_a.D_eff),
        'D_eff_c': float(p_c.D_eff),
        'R_a_m': float(p_a.R),
        'R_c_m': float(p_c.R),
        'voltage_labels_preserved': bool(rg_cfg.get('preserve_source_voltage_labels', True)),
        'phi_labels_preserved': bool(rg_cfg.get('preserve_source_phi_labels', True)),
        'mean_abs_grad_a_norm': float(np.mean(np.abs(out_arrays['grad_a_surface_center'])) / p_a.csmax_mol_m3),
        'mean_abs_grad_c_norm': float(np.mean(np.abs(out_arrays['grad_c_surface_center'])) / p_c.csmax_mol_m3),
    }
    write_json(summary, out_dir / 'soft_label_summary.json')
    return summary


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_dir)
    output_root = Path(args.output_dir)
    if output_root.exists() and any(output_root.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f'Output directory exists and is not empty: {output_root}. Use --allow-overwrite only for deliberate reruns.')
    output_root.mkdir(parents=True, exist_ok=True)
    prior = load_json(args.prior_json)
    files = discover_softlabel_npz(source_root, filename=args.filename)
    if not files:
        raise FileNotFoundError(f'No source {args.filename} found under {source_root}')
    summaries: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for idx, npz_path in enumerate(files, start=1):
        print(f'[D15-P0 RG generate] {idx}/{len(files)} {npz_path}', flush=True)
        try:
            summaries.append(_process_one(npz_path, source_root, output_root, prior, nr_override=args.nr))
        except Exception as exc:
            print(f'[D15-P0 RG generate] ERROR for {npz_path}: {exc!r}', flush=True)
            errors.append({'npz_path': str(npz_path), 'error': repr(exc)})
    report = {
        'stage': 'D15-P0 P2Dlite-RG 8-cell generation',
        'source_dir': str(source_root),
        'output_dir': str(output_root),
        'prior_json': str(args.prior_json),
        'source_profile_count': len(files),
        'generated_count': len(summaries),
        'error_count': len(errors),
        'overall_status': 'PASS' if len(errors) == 0 and len(summaries) == len(files) else 'FAIL',
        'summaries': summaries,
        'errors': errors,
    }
    write_json(report, output_root / 'D15_P0_RG_GENERATION_REPORT.json')
    with open(output_root / 'D15_P0_RG_GENERATION_REPORT.csv', 'w', encoding='utf-8-sig', newline='') as f:
        fieldnames = sorted({k for s in summaries for k in s.keys()}) if summaries else ['profile_id']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print('[D15-P0 RG generate] wrote:', output_root)
    print('[D15-P0 RG generate] overall_status:', report['overall_status'])
    return 0 if report['overall_status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
