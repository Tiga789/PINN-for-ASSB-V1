from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .g1_data import (
    G1Dataset,
    ProfilePack,
    json_dump,
    load_split_records,
    load_semantics_map,
    _semantics_for,
    load_profile_pack,
    save_profile_predictions,
)
from .g1_metrics import aggregate_profile_rows, profile_metrics, r2_score
from .g13_trainer import (
    _device_from_arg,
    _local_observed_features,
    _profile_summary_features,
    _replay_observed_aligned,
)
from .g14_model import ValidationRobustObservedProfileSurrogate


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return default


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fields.append(k); seen.add(k)
    if not fields:
        fields = ["empty"]
        rows = [{"empty": ""}]
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))


def torch_load_safe(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    # PyTorch 2.6 changed the default of weights_only in some builds.  The
    # checkpoint intentionally contains normalization arrays and config dicts,
    # so we request full loading when the keyword is supported.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location=map_location)


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def select_records(records: Sequence[Mapping[str, Any]], split: str, limit: int = 0, exclude_flagged: bool = True) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in records:
        if exclude_flagged and (bool(r.get("is_flagged_probe")) or str(r.get("split")) == "flagged_probe"):
            continue
        if str(r.get("split")) == split:
            out.append(dict(r))
    if limit and int(limit) > 0:
        out = out[: int(limit)]
    return out


def parse_vocab_from_feature_names(feature_names: Sequence[str], local_input_dim: int) -> Tuple[List[str], List[str], List[str]]:
    # In G1/G2, the base feature vector is followed by 7 absolute local observed
    # features and then profile summary features.  The base vector contains the
    # protocol_* and branch_* one-hot fields.  We parse these from the checkpoint
    # to guarantee that frozen-test features use the exact training convention.
    base_dim = max(0, int(local_input_dim) - 7)
    base_names = list(feature_names[:base_dim])
    protocols = [n[len("protocol_"):] for n in base_names if n.startswith("protocol_")]
    branches = [n[len("branch_"):] for n in base_names if n.startswith("branch_")]
    return base_names, protocols, branches


def semantic_for_record(record: Mapping[str, Any], sem_map: Mapping[str, Dict[str, str]]) -> Dict[str, str]:
    return _semantics_for(record, sem_map)


def load_profiles_for_split(
    records: Sequence[Mapping[str, Any]],
    split: str,
    sem_map: Mapping[str, Dict[str, str]],
    protocols: Sequence[str],
    branches: Sequence[str],
    max_time_points: int,
    time_window_s: float,
    limit: int = 0,
    exclude_flagged: bool = True,
) -> Tuple[List[ProfilePack], List[Dict[str, Any]]]:
    selected = select_records(records, split, limit=limit, exclude_flagged=exclude_flagged)
    profiles: List[ProfilePack] = []
    failures: List[Dict[str, Any]] = []
    for r in selected:
        try:
            p = load_profile_pack(r, semantic_for_record(r, sem_map), protocols, branches, max_time_points, time_window_s)
            profiles.append(p)
        except Exception as e:
            failures.append({
                "split": split,
                "canonical_cell_uid": r.get("canonical_cell_uid") or r.get("cell_uid"),
                "softlabel_npz": r.get("softlabel_npz"),
                "replay_npz": r.get("replay_npz"),
                "error": repr(e),
            })
    return profiles, failures


def augment_profile_features(profile: ProfilePack) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
    I, V, T, info = _replay_observed_aligned(profile)
    local_obs, local_obs_names = _local_observed_features(profile, I, V, T)
    pfeat, pnames = _profile_summary_features(profile, I, V, T)
    repeated = np.repeat(pfeat.reshape(1, -1), profile.features.shape[0], axis=0).astype(np.float32)
    X = np.concatenate([profile.features.astype(np.float32), local_obs.astype(np.float32), repeated], axis=1).astype(np.float32)
    feature_names = list(profile.feature_names) + local_obs_names + pnames
    info = dict(info)
    info.update({
        "canonical_cell_uid": profile.canonical_cell_uid,
        "split": profile.split,
        "protocol": profile.protocol,
        "semantic_branch": profile.branch,
        "n_time": int(profile.features.shape[0]),
        "feature_dim_augmented": int(X.shape[1]),
        "I_abs_max": float(np.nanmax(np.abs(I))) if I.size else float("nan"),
        "V_mean": float(np.nanmean(V)) if V.size else float("nan"),
    })
    return X, info, feature_names


class FrozenAuditData:
    def __init__(self, checkpoint: Mapping[str, Any], feature_audit_rows: Sequence[Mapping[str, Any]]):
        self.x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
        self.x_std = np.asarray(checkpoint["x_std"], dtype=np.float32)
        self.y_mean = np.asarray(checkpoint["y_mean"], dtype=np.float32)
        self.y_std = np.asarray(checkpoint["y_std"], dtype=np.float32)
        self.feature_names = list(checkpoint.get("feature_names") or [])
        self.profile_feature_names = list(checkpoint.get("profile_feature_names") or [])
        self.local_input_dim = int(checkpoint.get("local_input_dim", 0))
        self.profile_input_dim = int(checkpoint.get("profile_input_dim", 0))
        self.base = type("Base", (), {})()
        self.base.target_slices = {str(k): (int(v[0]), int(v[1])) for k, v in dict(checkpoint.get("target_slices") or {}).items()}
        self.feature_audit_rows = list(feature_audit_rows)


def predict_np(model: torch.nn.Module, X: np.ndarray, data: FrozenAuditData, device: torch.device, batch_size: int = 8192) -> np.ndarray:
    if X.shape[1] != data.x_mean.size:
        raise ValueError(f"feature dim mismatch: X has {X.shape[1]}, checkpoint x_mean has {data.x_mean.size}")
    xstd = data.x_std.copy()
    xstd[~np.isfinite(xstd) | (np.abs(xstd) < 1e-8)] = 1.0
    Xn = ((X - data.x_mean.reshape(1, -1)) / xstd.reshape(1, -1)).astype(np.float32)
    Xn[~np.isfinite(Xn)] = 0.0
    outs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, Xn.shape[0], int(batch_size)):
            xb = torch.as_tensor(Xn[i : i + int(batch_size)], dtype=torch.float32, device=device)
            outs.append(model(xb).detach().cpu().numpy())
    yn = np.concatenate(outs, axis=0) if outs else np.zeros((0, data.y_mean.size), dtype=np.float32)
    return (yn * data.y_std.reshape(1, -1) + data.y_mean.reshape(1, -1)).astype(np.float32)


def build_model_from_checkpoint(checkpoint: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    cfg = dict(checkpoint.get("config") or {})
    model_cfg = dict(cfg.get("model") or {})
    target_slices = {str(k): (int(v[0]), int(v[1])) for k, v in dict(checkpoint.get("target_slices") or {}).items()}
    model = ValidationRobustObservedProfileSurrogate(
        local_input_dim=int(checkpoint.get("local_input_dim")),
        profile_input_dim=int(checkpoint.get("profile_input_dim", 0)),
        target_slices=target_slices,
        width=int(model_cfg.get("width", 960)),
        depth=int(model_cfg.get("depth", 8)),
        profile_width=int(model_cfg.get("profile_width", 288)),
        dropout=float(model_cfg.get("dropout", 0.05)),
        phie_direct_width=int(model_cfg.get("phie_direct_width", 384)),
    ).to(device)
    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state is None:
        raise KeyError("checkpoint does not contain model_state_dict/state_dict")
    model.load_state_dict(state)
    return model


def per_target_rows(profiles: Sequence[ProfilePack], preds: Sequence[np.ndarray], split_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for prof, pred in zip(profiles, preds):
        for key, (a, b) in prof.target_slices.items():
            yt = prof.targets[:, a:b]
            yp = pred[:, a:b]
            rows.append({
                "split": split_name,
                "canonical_cell_uid": prof.canonical_cell_uid,
                "cell_uid": prof.cell_uid,
                "protocol": prof.protocol,
                "semantic_branch": prof.branch,
                "target": key,
                "mae": float(np.nanmean(np.abs(yp - yt))),
                "rmse": float(np.sqrt(np.nanmean((yp - yt) ** 2))),
                "r2": r2_score(yt, yp),
                "n_points": int(yt.size),
                "target_range": float(np.nanmax(yt) - np.nanmin(yt)) if yt.size else float("nan"),
                "target_std": float(np.nanstd(yt)) if yt.size else float("nan"),
                "bias": float(np.nanmean(yp - yt)) if yt.size else float("nan"),
            })
    return rows


def target_aggregate(rows: Sequence[Mapping[str, Any]], split_name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"split": split_name}
    vals: List[float] = []
    for target in sorted({str(r.get("target")) for r in rows}):
        rs = [safe_float(r.get("r2")) for r in rows if str(r.get("target")) == target]
        rs = [v for v in rs if math.isfinite(v)]
        if rs:
            out[f"{target}_r2_mean"] = float(np.mean(rs))
            out[f"{target}_r2_min"] = float(np.min(rs))
            vals.extend(rs)
    out["all_target_profile_r2_mean"] = float(np.mean(vals)) if vals else float("nan")
    out["all_target_profile_r2_min"] = float(np.min(vals)) if vals else float("nan")
    return out


def profile_aggregate(profiles: Sequence[ProfilePack], preds: Sequence[np.ndarray], split_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not profiles:
        return [], {"split": split_name}
    rows = profile_metrics(profiles, preds)["rows"]
    for r in rows:
        r["split"] = split_name
    agg = aggregate_profile_rows(rows)
    agg["split"] = split_name
    return rows, agg


def phie_rows(profiles: Sequence[ProfilePack], preds: Sequence[np.ndarray], split_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for prof, pred in zip(profiles, preds):
        if "phie" not in prof.target_slices:
            continue
        a, b = prof.target_slices["phie"]
        yt = prof.targets[:, a:b]
        yp = pred[:, a:b]
        rows.append({
            "split": split_name,
            "canonical_cell_uid": prof.canonical_cell_uid,
            "protocol": prof.protocol,
            "semantic_branch": prof.branch,
            "phie_r2": r2_score(yt, yp),
            "phie_mae": float(np.nanmean(np.abs(yp - yt))),
            "phie_rmse": float(np.sqrt(np.nanmean((yp - yt) ** 2))),
            "phie_bias": float(np.nanmean(yp - yt)),
            "phie_true_range": float(np.nanmax(yt) - np.nanmin(yt)),
            "phie_true_std": float(np.nanstd(yt)),
            "phie_pred_range": float(np.nanmax(yp) - np.nanmin(yp)),
            "phie_pred_std": float(np.nanstd(yp)),
            "n_time": int(yt.shape[0]),
        })
    return rows


def worst_row(rows: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    finite_rows = [dict(r) for r in rows if math.isfinite(safe_float(r.get("r2")))]
    if not finite_rows:
        return None
    return min(finite_rows, key=lambda r: safe_float(r.get("r2"), 1e99))


def group_counts(profiles: Sequence[ProfilePack], attr: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for p in profiles:
        if attr == "protocol_branch":
            k = f"{p.protocol}::{p.branch}"
        else:
            k = str(getattr(p, attr, "UNKNOWN") or "UNKNOWN")
        out[k] = out.get(k, 0) + 1
    return out


def load_candidate_summary(path: str | Path) -> Dict[str, Any]:
    d = read_json(path, default={}) or {}
    return dict(d) if isinstance(d, Mapping) else {}


def resolve_checkpoint_path(args_checkpoint: str, candidate_summary: Mapping[str, Any], candidate_dir: str | Path) -> Path:
    if args_checkpoint:
        return Path(args_checkpoint)
    files = candidate_summary.get("files") if isinstance(candidate_summary.get("files"), Mapping) else {}
    if files and files.get("best_model_pt"):
        return Path(str(files.get("best_model_pt")))
    return Path(candidate_dir) / "model" / "best_model.pt"


def run_g3_frozen_test_audit(
    split_manifest: str | Path,
    g0_profile_semantics_csv: str | Path,
    candidate_g21_dir: str | Path,
    candidate_g21_summary: str | Path,
    out_dir: str | Path,
    config: Mapping[str, Any],
    checkpoint_path: str | Path = "",
    max_time_points: int = 512,
    time_window_s: float = 40000.0,
    frozen_test_profile_limit: int = 0,
    flagged_probe_profile_limit: int = 1,
    device_arg: str = "auto",
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidate = load_candidate_summary(candidate_g21_summary)
    candidate_ready = bool(candidate.get("g3_ready")) and str(candidate.get("status")) == "PASS"
    if bool(config.get("require_g21_ready", True)) and not candidate_ready:
        summary = {
            "protocol": "D17-G3_FROZEN_TEST_REPORT_ONLY_AUDIT",
            "created_at_utc": utc_now(),
            "status": "BLOCKED",
            "promotion_status": "BLOCKED",
            "g4_ready": False,
            "blockers": ["candidate G2.1 summary is not PASS/g3_ready=true"],
            "candidate_g21_summary": str(candidate_g21_summary),
            "candidate_status": candidate.get("status"),
            "candidate_g3_ready": candidate.get("g3_ready"),
        }
        json_dump(summary, out / "D17_G3_FROZEN_TEST_REPORT_ONLY_AUDIT_SUMMARY.json")
        return summary

    ckpt_path = resolve_checkpoint_path(str(checkpoint_path or ""), candidate, candidate_g21_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Cannot find G2.1 checkpoint: {ckpt_path}")
    ckpt = torch_load_safe(ckpt_path, map_location="cpu")
    device = _device_from_arg(device_arg)
    model = build_model_from_checkpoint(ckpt, device)

    feature_names = list(ckpt.get("feature_names") or [])
    local_input_dim = int(ckpt.get("local_input_dim", 0))
    base_feature_names, protocol_vocab, branch_vocab = parse_vocab_from_feature_names(feature_names, local_input_dim)
    if not protocol_vocab or not branch_vocab:
        raise ValueError("Could not parse protocol/branch vocab from checkpoint feature_names")

    records, manifest = load_split_records(split_manifest)
    sem_map = load_semantics_map(g0_profile_semantics_csv)
    frozen_profiles, frozen_failures = load_profiles_for_split(
        records,
        "frozen_test",
        sem_map,
        protocol_vocab,
        branch_vocab,
        max_time_points=int(max_time_points),
        time_window_s=float(time_window_s),
        limit=int(frozen_test_profile_limit),
        exclude_flagged=True,
    )
    flagged_profiles, flagged_failures = load_profiles_for_split(
        records,
        "flagged_probe",
        sem_map,
        protocol_vocab,
        branch_vocab,
        max_time_points=int(max_time_points),
        time_window_s=float(time_window_s),
        limit=int(flagged_probe_profile_limit),
        exclude_flagged=False,
    )

    feature_audit_rows: List[Dict[str, Any]] = []
    all_profile_X: Dict[str, np.ndarray] = {}
    feature_mismatch_rows: List[Dict[str, Any]] = []
    for split_name, profiles in [("frozen_test", frozen_profiles), ("flagged_probe", flagged_profiles)]:
        for p in profiles:
            X, info, aug_names = augment_profile_features(p)
            info["audit_split"] = split_name
            feature_audit_rows.append(info)
            all_profile_X[p.canonical_cell_uid] = X
            if list(aug_names) != feature_names:
                feature_mismatch_rows.append({
                    "split": split_name,
                    "canonical_cell_uid": p.canonical_cell_uid,
                    "aug_feature_dim": len(aug_names),
                    "checkpoint_feature_dim": len(feature_names),
                    "same_feature_names": False,
                    "first_aug_only": next((n for n, m in zip(aug_names, feature_names) if n != m), ""),
                })
    audit_data = FrozenAuditData(ckpt, feature_audit_rows)

    def predict_profiles(profiles: Sequence[ProfilePack]) -> List[np.ndarray]:
        preds: List[np.ndarray] = []
        for p in profiles:
            preds.append(predict_np(model, all_profile_X[p.canonical_cell_uid], audit_data, device, batch_size=int(config.get("predict_batch_size", 8192))))
        return preds

    frozen_preds = predict_profiles(frozen_profiles)
    flagged_preds = predict_profiles(flagged_profiles) if flagged_profiles else []

    frozen_profile_rows, frozen_profile_agg = profile_aggregate(frozen_profiles, frozen_preds, "frozen_test_report_only")
    frozen_trows = per_target_rows(frozen_profiles, frozen_preds, "frozen_test_report_only")
    frozen_tagg = target_aggregate(frozen_trows, "frozen_test_report_only")
    flagged_profile_rows, flagged_profile_agg = profile_aggregate(flagged_profiles, flagged_preds, "flagged_probe_report_only")
    flagged_trows = per_target_rows(flagged_profiles, flagged_preds, "flagged_probe_report_only")
    flagged_tagg = target_aggregate(flagged_trows, "flagged_probe_report_only")
    all_phie_rows = phie_rows(frozen_profiles, frozen_preds, "frozen_test_report_only") + phie_rows(flagged_profiles, flagged_preds, "flagged_probe_report_only")

    pred_manifest: List[Dict[str, Any]] = []
    pred_manifest.extend(save_profile_predictions(out / "predictions" / "frozen_test", "G3_frozen_test_report_only", frozen_profiles, frozen_preds))
    if flagged_profiles:
        pred_manifest.extend(save_profile_predictions(out / "predictions" / "flagged_probe", "G3_flagged_probe_report_only", flagged_profiles, flagged_preds))

    write_csv(frozen_profile_rows + flagged_profile_rows, out / "D17_G3_PROFILE_METRICS.csv")
    write_csv(frozen_trows + flagged_trows, out / "D17_G3_PER_TARGET_PROFILE_METRICS.csv")
    write_csv([frozen_tagg, flagged_tagg], out / "D17_G3_PER_TARGET_AGGREGATE.csv")
    write_csv(all_phie_rows, out / "D17_G3_PHIE_AUDIT.csv")
    write_csv(pred_manifest, out / "D17_G3_PREDICTION_MANIFEST.csv")
    write_csv(feature_audit_rows, out / "D17_G3_PROFILE_ENCODER_FEATURE_AUDIT.csv")
    write_csv(feature_mismatch_rows, out / "D17_G3_FEATURE_NAME_MISMATCH_AUDIT.csv")
    write_csv(frozen_failures + flagged_failures, out / "D17_G3_LOAD_FAILURES.csv")

    frozen_mean = safe_float(frozen_tagg.get("all_target_profile_r2_mean"))
    frozen_min = safe_float(frozen_tagg.get("all_target_profile_r2_min"))
    frozen_phie_mean = safe_float(frozen_tagg.get("phie_r2_mean"))
    frozen_phie_min = safe_float(frozen_tagg.get("phie_r2_min"))
    blockers: List[str] = []
    status_reasons: List[str] = []
    if frozen_failures:
        status_reasons.append(f"{len(frozen_failures)} frozen_test profiles failed to load")
    if feature_mismatch_rows:
        status_reasons.append("feature names for one or more frozen/flagged profiles do not exactly match checkpoint feature_names")
    if not frozen_profiles:
        status_reasons.append("no frozen_test profiles were evaluated")
    status = "PASS" if not status_reasons else "REVIEW"

    if frozen_mean < float(config.get("frozen_r2_mean_threshold", 0.95)) or frozen_min < float(config.get("frozen_r2_min_threshold", 0.90)):
        blockers.append(f"frozen-test target/profile R2 below gate: mean={frozen_mean:.6g}, min={frozen_min:.6g}")
    if frozen_phie_mean < float(config.get("frozen_phie_r2_mean_threshold", 0.93)) or frozen_phie_min < float(config.get("frozen_phie_r2_min_threshold", 0.90)):
        blockers.append(f"frozen-test phie R2 below gate: mean={frozen_phie_mean:.6g}, min={frozen_phie_min:.6g}")
    promotion_status = "PASS" if status == "PASS" and not blockers else "REVIEW"
    g4_ready = bool(promotion_status == "PASS")
    recommendation = "ENTER_D17_G4_FINAL_SCORECARD_AND_SPEED_AUDIT" if g4_ready else "DO_NOT_ENTER_G4_REVIEW_FROZEN_TEST_FAILURES"

    freeze_manifest = {
        "candidate_protocol": candidate.get("protocol"),
        "candidate_status": candidate.get("status"),
        "candidate_g3_ready": candidate.get("g3_ready"),
        "candidate_summary": str(candidate_g21_summary),
        "checkpoint": str(ckpt_path),
        "checkpoint_best_epoch": ckpt.get("best_epoch"),
        "split_manifest": str(split_manifest),
        "split_manifest_hash_sha256": manifest.get("manifest_hash_sha256"),
        "g0_profile_semantics_csv": str(g0_profile_semantics_csv),
        "created_at_utc": utc_now(),
        "frozen_test_softlabels_used_for_training": False,
        "checkpoint_selection_performed": False,
        "training_performed": False,
    }
    json_dump(freeze_manifest, out / "D17_G3_FREEZE_MANIFEST.json")

    summary: Dict[str, Any] = {
        "protocol": "D17-G3_FROZEN_TEST_REPORT_ONLY_AUDIT",
        "created_at_utc": utc_now(),
        "status": status,
        "status_reasons": status_reasons,
        "promotion_status": promotion_status,
        "g4_ready": bool(g4_ready),
        "recommendation": recommendation,
        "g4_blockers": blockers,
        "purpose": "One-time frozen-test report-only audit of the D17-G2.1 supervised generator-surrogate candidate. No training, checkpoint selection, split editing, or frozen-test feedback is performed.",
        "policy": {
            "train_cell_softlabels_used_in_upstream_G2_1_training": True,
            "validation_softlabels_were_report_only_in_G2_1": True,
            "frozen_test_softlabels_used_for_training": False,
            "frozen_test_softlabels_used_for_checkpoint_selection": False,
            "training_performed_in_G3": False,
            "checkpoint_selection_performed_in_G3": False,
            "softlabels_read_stage": "report_only_metric_after_loading_frozen_profiles; model inputs remain observed I/V/T/protocol/branch features",
        },
        "device": str(device),
        "dataset": {
            "manifest_hash_sha256": manifest.get("manifest_hash_sha256"),
            "record_counts": manifest.get("counts"),
            "frozen_test_profile_count": len(frozen_profiles),
            "flagged_probe_profile_count": len(flagged_profiles),
            "frozen_test_protocol_counts": group_counts(frozen_profiles, "protocol"),
            "frozen_test_semantic_branch_counts": group_counts(frozen_profiles, "branch"),
            "frozen_test_protocol_branch_counts": group_counts(frozen_profiles, "protocol_branch"),
            "flagged_probe_protocol_counts": group_counts(flagged_profiles, "protocol"),
            "checkpoint_protocol_vocab": protocol_vocab,
            "checkpoint_semantic_branch_vocab": branch_vocab,
            "target_dim": int(np.asarray(ckpt.get("y_mean")).size),
            "augmented_feature_dim": int(np.asarray(ckpt.get("x_mean")).size),
        },
        "candidate": freeze_manifest,
        "frozen_test_per_target_aggregate": frozen_tagg,
        "frozen_test_profile_aggregate": frozen_profile_agg,
        "flagged_probe_per_target_aggregate": flagged_tagg,
        "flagged_probe_profile_aggregate": flagged_profile_agg,
        "worst_frozen_test_target_profile": worst_row(frozen_trows),
        "worst_flagged_probe_target_profile": worst_row(flagged_trows),
        "load_failures": frozen_failures + flagged_failures,
        "feature_name_mismatch_count": len(feature_mismatch_rows),
        "files": {
            "summary_json": str(out / "D17_G3_FROZEN_TEST_REPORT_ONLY_AUDIT_SUMMARY.json"),
            "freeze_manifest_json": str(out / "D17_G3_FREEZE_MANIFEST.json"),
            "profile_metrics_csv": str(out / "D17_G3_PROFILE_METRICS.csv"),
            "per_target_profile_metrics_csv": str(out / "D17_G3_PER_TARGET_PROFILE_METRICS.csv"),
            "per_target_aggregate_csv": str(out / "D17_G3_PER_TARGET_AGGREGATE.csv"),
            "phie_audit_csv": str(out / "D17_G3_PHIE_AUDIT.csv"),
            "prediction_manifest_csv": str(out / "D17_G3_PREDICTION_MANIFEST.csv"),
            "feature_audit_csv": str(out / "D17_G3_PROFILE_ENCODER_FEATURE_AUDIT.csv"),
            "feature_mismatch_audit_csv": str(out / "D17_G3_FEATURE_NAME_MISMATCH_AUDIT.csv"),
            "load_failures_csv": str(out / "D17_G3_LOAD_FAILURES.csv"),
        },
    }
    json_dump(summary, out / "D17_G3_FROZEN_TEST_REPORT_ONLY_AUDIT_SUMMARY.json")
    # scorecard is intentionally a compact alias for downstream inspection.
    scorecard = {
        "protocol": summary["protocol"],
        "status": summary["status"],
        "promotion_status": summary["promotion_status"],
        "g4_ready": summary["g4_ready"],
        "g4_blockers": blockers,
        "frozen_test_per_target_aggregate": frozen_tagg,
        "worst_frozen_test_target_profile": summary["worst_frozen_test_target_profile"],
        "policy": summary["policy"],
    }
    json_dump(scorecard, out / "D17_G3_SCORECARD.json")
    return summary
