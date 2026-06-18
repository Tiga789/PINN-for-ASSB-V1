from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d17_g.g6f_selected_cycle_infer import (
    OBS_TIME_KEYS,
    _find_key,
    _interp_obs_from_replay,
    _as_time_radial,
    _to_1d,
    build_augmented_features,
    build_model_from_checkpoint,
    device_from_arg,
    find_record,
    load_candidate_checkpoint,
    load_semantics_map,
    load_split_records,
    metrics_for_target,
    parse_vocabs_from_checkpoint_feature_names,
    predict_array,
    semantics_for_record,
    slice_prediction_by_targets,
)


OBS_I_KEYS = ["I_profile", "current_A", "I_A", "current", "I"]
OBS_V_KEYS = ["voltage_exp", "voltage_V", "V_exp", "V", "voltage"]
OBS_T_KEYS = ["temperature_C", "temp_C", "T_C", "temperature_K", "T", "temperature"]


def scalar_str(z: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in z.files:
        return default
    try:
        arr = np.asarray(z[key])
        if arr.shape == ():
            v = arr.item()
        else:
            v = arr.reshape(-1)[0]
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        return str(v)
    except Exception:
        return default


def nearest_indices(source_t: np.ndarray, query_t: np.ndarray) -> tuple[np.ndarray, float, float]:
    source_t = np.asarray(source_t, dtype=np.float64).reshape(-1)
    query_t = np.asarray(query_t, dtype=np.float64).reshape(-1)
    order = np.argsort(source_t)
    st = source_t[order]
    pos = np.searchsorted(st, query_t)
    pos0 = np.clip(pos - 1, 0, st.size - 1)
    pos1 = np.clip(pos, 0, st.size - 1)
    choose1 = np.abs(st[pos1] - query_t) < np.abs(st[pos0] - query_t)
    chosen = np.where(choose1, pos1, pos0)
    idx = order[chosen]
    err = np.abs(source_t[idx] - query_t)
    return idx.astype(np.int64), float(np.median(err)), float(np.max(err))


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    if not keys:
        keys = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g3_pred_npz", required=True)
    ap.add_argument("--split_manifest", required=True)
    ap.add_argument("--g0_profile_semantics_csv", required=True)
    ap.add_argument("--candidate_dir", required=True)
    ap.add_argument("--candidate_summary", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--battery", required=True)
    ap.add_argument("--targets", nargs="+", default=["cs_a", "cs_c", "phie", "phis_c"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--predict_batch_size", type=int, default=8192)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g3_pred_npz = Path(args.g3_pred_npz)
    if not g3_pred_npz.exists():
        raise FileNotFoundError(g3_pred_npz)

    records, manifest = load_split_records(args.split_manifest)
    record = find_record(records, str(args.batch), str(args.battery))
    sem_map = load_semantics_map(args.g0_profile_semantics_csv)
    sem = semantics_for_record(record, sem_map)

    device = device_from_arg(args.device)
    ckpt, ckpt_path, candidate_summary = load_candidate_checkpoint(
        args.candidate_dir,
        args.candidate_summary,
        args.checkpoint,
    )
    model = build_model_from_checkpoint(ckpt, device)

    feature_names_ckpt = list(ckpt.get("feature_names") or [])
    target_slices = {
        str(k): (int(v[0]), int(v[1]))
        for k, v in dict(ckpt.get("target_slices") or {}).items()
    }
    protocol_vocab, branch_vocab = parse_vocabs_from_checkpoint_feature_names(
        feature_names_ckpt,
        int(ckpt.get("local_input_dim", 0)),
    )

    with np.load(g3_pred_npz, allow_pickle=True) as gz:
        if "t_global_s" not in gz.files:
            raise KeyError("G3 pred npz has no t_global_s")
        t = np.asarray(gz["t_global_s"], dtype=np.float32).reshape(-1)
        n_time = int(t.size)

        protocol = scalar_str(gz, "protocol", str(record.get("protocol") or "UNKNOWN"))
        branch = scalar_str(gz, "semantic_branch", str(sem.get("semantic_branch") or "UNKNOWN_OR_MIXED_BRANCH"))

        g3_pred: Dict[str, np.ndarray] = {}
        g3_true: Dict[str, np.ndarray] = {}
        for target in args.targets:
            pk = f"{target}_pred"
            tk = f"{target}_true_report_only"
            if pk not in gz.files or tk not in gz.files:
                raise KeyError(f"Missing {pk} or {tk} in G3 pred npz")
            g3_pred[target] = _as_time_radial(gz[pk], n_time, pk)
            g3_true[target] = _as_time_radial(gz[tk], n_time, tk)

    soft_npz = Path(str(record.get("softlabel_npz") or ""))
    replay_npz = Path(str(record.get("replay_npz") or ""))

    step_type = None
    soft_nearest_median_error_s = None
    soft_nearest_max_error_s = None

    if soft_npz.exists():
        with np.load(soft_npz, allow_pickle=True) as sz:
            stk = _find_key(sz, OBS_TIME_KEYS)
            if stk is not None:
                st_full = _to_1d(sz[stk])
                idx, med_err, max_err = nearest_indices(st_full, t)
                soft_nearest_median_error_s = med_err
                soft_nearest_max_error_s = max_err
                if "step_type" in sz.files:
                    st = np.asarray(sz["step_type"]).reshape(-1)
                    if st.size == st_full.size:
                        step_type = st[idx]

    I_src, I = _interp_obs_from_replay(replay_npz, t, OBS_I_KEYS, 0.0)
    V_src, V = _interp_obs_from_replay(replay_npz, t, OBS_V_KEYS, 0.0)
    T_src, T = _interp_obs_from_replay(replay_npz, t, OBS_T_KEYS, 25.0)

    X, feature_names = build_augmented_features(
        t,
        I.astype(np.float32),
        V.astype(np.float32),
        T.astype(np.float32),
        step_type,
        protocol,
        branch,
        protocol_vocab,
        branch_vocab,
    )

    feature_match = list(feature_names) == list(feature_names_ckpt)
    first_feature_mismatch = None
    if not feature_match:
        for i, (a, b) in enumerate(zip(feature_names, feature_names_ckpt)):
            if a != b:
                first_feature_mismatch = {"index": i, "g6f_feature": a, "ckpt_feature": b}
                break
        if first_feature_mismatch is None and len(feature_names) != len(feature_names_ckpt):
            first_feature_mismatch = {
                "g6f_feature_count": len(feature_names),
                "ckpt_feature_count": len(feature_names_ckpt),
            }

    if X.shape[1] != np.asarray(ckpt["x_mean"]).size:
        raise ValueError(
            f"Feature dimension mismatch: X={X.shape[1]} checkpoint={np.asarray(ckpt['x_mean']).size}"
        )

    pred_full = predict_array(model, X, ckpt, device, batch_size=int(args.predict_batch_size))
    g6f_pred = slice_prediction_by_targets(pred_full, target_slices, args.targets)

    rows: List[Dict[str, Any]] = []
    summary_targets: Dict[str, Any] = {}

    for target in args.targets:
        m_g3_true = metrics_for_target(g3_true[target], g3_pred[target])
        m_g6f_g3 = metrics_for_target(g3_pred[target], g6f_pred[target])
        m_g6f_true = metrics_for_target(g3_true[target], g6f_pred[target])

        summary_targets[target] = {
            "g3_saved_vs_true": m_g3_true,
            "g6f_vs_g3_saved": m_g6f_g3,
            "g6f_vs_true_on_g3_grid": m_g6f_true,
        }

        for level, mm in [
            ("g3_saved_vs_true", m_g3_true),
            ("g6f_vs_g3_saved", m_g6f_g3),
            ("g6f_vs_true_on_g3_grid", m_g6f_true),
        ]:
            rows.append({
                "level": level,
                "target": target,
                **mm,
            })

    min_g6f_vs_g3 = min(
        float(summary_targets[t]["g6f_vs_g3_saved"]["r2"])
        for t in args.targets
    )
    mean_g6f_vs_g3 = float(np.mean([
        float(summary_targets[t]["g6f_vs_g3_saved"]["r2"])
        for t in args.targets
    ]))
    min_g6f_vs_true = min(
        float(summary_targets[t]["g6f_vs_true_on_g3_grid"]["r2"])
        for t in args.targets
    )
    mean_g6f_vs_true = float(np.mean([
        float(summary_targets[t]["g6f_vs_true_on_g3_grid"]["r2"])
        for t in args.targets
    ]))

    if min_g6f_vs_g3 >= 0.999:
        recommendation = "G6F_FORWARD_MATCHES_G3_ON_SAVED_GRID_DENSE_FAILURE_IS_MODEL_DOMAIN_OR_GRID_GENERALIZATION"
        sanity_status = "PASS"
    else:
        recommendation = "G6F_FORWARD_DOES_NOT_MATCH_G3_ON_SAVED_GRID_FIX_G6F_FEATURE_OR_NORMALIZATION_PATH"
        sanity_status = "REVIEW"

    summary = {
        "protocol": "D17-G6F_EXACT_GRID_SANITY_AGAINST_G3",
        "status": "PASS",
        "sanity_status": sanity_status,
        "recommendation": recommendation,
        "g3_pred_npz": str(g3_pred_npz),
        "candidate_checkpoint": str(ckpt_path),
        "cell": str(record.get("canonical_cell_uid") or record.get("cell_uid")),
        "n_time": n_time,
        "time_range": [float(np.min(t)), float(np.max(t))],
        "protocol_value": protocol,
        "semantic_branch_value": branch,
        "observed_sources": {
            "I": I_src,
            "V": V_src,
            "T": T_src,
        },
        "softlabel_time_alignment": {
            "soft_npz": str(soft_npz),
            "nearest_median_abs_error_s": soft_nearest_median_error_s,
            "nearest_max_abs_error_s": soft_nearest_max_error_s,
        },
        "feature_check": {
            "feature_match": feature_match,
            "first_feature_mismatch": first_feature_mismatch,
            "feature_dim": int(X.shape[1]),
            "checkpoint_feature_dim": int(np.asarray(ckpt["x_mean"]).size),
        },
        "aggregate": {
            "g6f_vs_g3_saved_r2_mean": mean_g6f_vs_g3,
            "g6f_vs_g3_saved_r2_min": min_g6f_vs_g3,
            "g6f_vs_true_on_g3_grid_r2_mean": mean_g6f_vs_true,
            "g6f_vs_true_on_g3_grid_r2_min": min_g6f_vs_true,
        },
        "target_metrics": summary_targets,
    }

    summary_path = out_dir / "D17_G6F_EXACT_GRID_SANITY_SUMMARY.json"
    metrics_csv = out_dir / "D17_G6F_EXACT_GRID_SANITY_METRICS.csv"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_csv(rows, metrics_csv)

    print(json.dumps({
        "status": summary["status"],
        "sanity_status": summary["sanity_status"],
        "recommendation": summary["recommendation"],
        "aggregate": summary["aggregate"],
        "feature_check": summary["feature_check"],
        "summary_json": str(summary_path),
        "metrics_csv": str(metrics_csv),
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
