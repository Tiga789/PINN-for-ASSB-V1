# -*- coding: utf-8 -*-
"""Strict 30/70 split utilities for ASSB ModelFin_111.

ModelFin_111 is intentionally different from ModelFin_110 Stage-B: the
SOH predictor may see only the visible 30% capacity/SOH labels. The held-out
70% cycles are used for final prediction evaluation and must never enter loss,
scaler fitting, early stopping, or model selection.

Main convention
---------------
Complete-cycle main range: cycle 5..521, inclusive.
Visible-label range:       cycle 5..159, inclusive.
Train range:               cycle 5..139, inclusive.
Validation range:          cycle 140..159, inclusive.
Held-out test range:       cycle 160..521, inclusive.
Partial/report-only:       cycle 522.

This module is standalone and safe to import from scripts. It avoids changing
any ModelFin_107A core file.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union, Any
import hashlib
import json
import math

import numpy as np
import pandas as pd

try:  # Reuse current repository capacity standardization when available.
    from util.assb_aging_capacity import standardize_capacity_targets
except Exception:  # pragma: no cover
    standardize_capacity_targets = None  # type: ignore

PathLike = Union[str, Path]


@dataclass(frozen=True)
class Strict30SplitConfig:
    """Canonical ModelFin_111 strict-30 split definition."""

    model_id: int = 111
    complete_cycle_min: int = 5
    complete_cycle_max: int = 521
    visible_label_cycle_min: int = 5
    visible_label_cycle_max: int = 159
    train_cycle_min: int = 5
    train_cycle_max: int = 139
    val_cycle_min: int = 140
    val_cycle_max: int = 159
    test_cycle_min: int = 160
    test_cycle_max: int = 521
    partial_cycles: Tuple[int, ...] = (522,)

    @property
    def forbidden_fit_splits(self) -> Tuple[str, ...]:
        return ("test", "partial")

    @property
    def train_cycles(self) -> Tuple[int, int]:
        return (self.train_cycle_min, self.train_cycle_max)

    @property
    def val_cycles(self) -> Tuple[int, int]:
        return (self.val_cycle_min, self.val_cycle_max)

    @property
    def test_cycles(self) -> Tuple[int, int]:
        return (self.test_cycle_min, self.test_cycle_max)

    @property
    def visible_label_cycles(self) -> Tuple[int, int]:
        return (self.visible_label_cycle_min, self.visible_label_cycle_max)


def _to_int_set(values: Optional[Iterable[int]]) -> set[int]:
    if values is None:
        return set()
    return {int(v) for v in values}


def _range_set(pair: Sequence[int]) -> set[int]:
    a, b = int(pair[0]), int(pair[1])
    if b < a:
        return set()
    return set(range(a, b + 1))


def sha256_file(path: PathLike) -> str:
    """Return SHA256 for provenance; empty string when the file is absent."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_jsonable(obj: Any) -> str:
    text = json.dumps(_json_clean(obj), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_clean(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_json_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_json_clean(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        val = float(x)
        return None if not math.isfinite(val) else val
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    return x


def _fallback_standardize_capacity_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Small fallback when util.assb_aging_capacity is unavailable."""
    def norm(s: str) -> str:
        return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())

    cols = {norm(c): c for c in frame.columns}

    def find(candidates: Sequence[str]) -> Optional[str]:
        for c in candidates:
            if norm(c) in cols:
                return cols[norm(c)]
        return None

    cycle_col = find(["cycle_id", "cycle", "循环", "循环号"])
    q_col = find([
        "Q_obs_Ah", "Q_obs_mAh", "Q_dis_Ah", "Q_dis_mAh",
        "放电容量(Ah)", "放电容量（Ah）", "放电容量(mAh)", "capacity_Ah", "capacity_mAh",
    ])
    soh_col = find(["SOH_obs", "SOH", "soh_target", "capacity_norm"])
    complete_col = find(["complete_cycle", "complete", "is_complete"])
    if cycle_col is None:
        raise KeyError(f"Cannot locate cycle column. Available columns: {list(frame.columns)}")

    out = pd.DataFrame()
    out["cycle_id"] = pd.to_numeric(frame[cycle_col], errors="coerce").astype("Int64")
    if q_col is not None:
        q_raw = pd.to_numeric(frame[q_col], errors="coerce").to_numpy(dtype=float)
        name = str(q_col).lower()
        finite = q_raw[np.isfinite(q_raw)]
        q_is_mAh = ("mah" in name) or (finite.size > 0 and np.nanmedian(np.abs(finite)) > 0.02)
        q_ah = q_raw / 1000.0 if q_is_mAh else q_raw
    else:
        q_ah = np.full(len(frame), np.nan)
    out["Q_obs_Ah"] = q_ah
    out["Q_obs_mAh"] = q_ah * 1000.0
    if soh_col is not None:
        out["SOH_obs"] = pd.to_numeric(frame[soh_col], errors="coerce").to_numpy(dtype=float)
    else:
        ref = np.nanmax(q_ah) if np.isfinite(q_ah).any() else np.nan
        out["SOH_obs"] = q_ah / ref if np.isfinite(ref) and ref > 0 else np.nan
    if complete_col is not None:
        out["complete_cycle"] = frame[complete_col].map(
            lambda x: str(x).strip().lower() not in {"0", "false", "no", "nan", "none", ""}
        ).astype(bool)
    else:
        out["complete_cycle"] = np.isfinite(out["Q_obs_Ah"].to_numpy(dtype=float))
    out = out.dropna(subset=["cycle_id"]).copy()
    out["cycle_id"] = out["cycle_id"].astype(int)
    return out.sort_values("cycle_id").drop_duplicates("cycle_id", keep="last").reset_index(drop=True)


def load_capacity_targets_strict30(
    capacity_target_csv: PathLike,
    *,
    cycle_from: int = 5,
    cycle_to: int = 522,
) -> pd.DataFrame:
    """Load capacity/SOH targets and canonicalize historical column names."""
    path = Path(capacity_target_csv)
    if not path.exists():
        raise FileNotFoundError(f"capacity target CSV not found: {path}")
    raw = pd.read_csv(path)
    if standardize_capacity_targets is not None:
        frame = standardize_capacity_targets(raw)  # type: ignore[misc]
    else:  # pragma: no cover
        frame = _fallback_standardize_capacity_targets(raw)
    frame = frame[(frame["cycle_id"] >= int(cycle_from)) & (frame["cycle_id"] <= int(cycle_to))]
    if frame.empty:
        raise RuntimeError(f"No capacity targets after filtering cycles {cycle_from}..{cycle_to}: {path}")
    return frame.reset_index(drop=True)


def assign_split(cycle_id: int, cfg: Strict30SplitConfig = Strict30SplitConfig()) -> str:
    c = int(cycle_id)
    if c in set(cfg.partial_cycles):
        return "partial"
    if cfg.train_cycle_min <= c <= cfg.train_cycle_max:
        return "train"
    if cfg.val_cycle_min <= c <= cfg.val_cycle_max:
        return "val"
    if cfg.test_cycle_min <= c <= cfg.test_cycle_max:
        return "test"
    if cfg.complete_cycle_min <= c <= cfg.complete_cycle_max:
        # Should not normally occur, but keeps future alternate splits explicit.
        return "complete_unassigned"
    return "out_of_scope"


def build_split_frame(
    targets: Optional[pd.DataFrame] = None,
    cfg: Strict30SplitConfig = Strict30SplitConfig(),
) -> pd.DataFrame:
    """Return one row per cycle with split and visibility flags."""
    cycles = list(range(int(cfg.complete_cycle_min), int(cfg.complete_cycle_max) + 1)) + list(cfg.partial_cycles)
    frame = pd.DataFrame({"cycle_id": sorted(set(cycles))})
    frame["split"] = frame["cycle_id"].map(lambda x: assign_split(int(x), cfg))
    frame["is_train"] = frame["split"].eq("train")
    frame["is_val"] = frame["split"].eq("val")
    frame["is_test"] = frame["split"].eq("test")
    frame["is_partial"] = frame["split"].eq("partial")
    frame["label_visible"] = frame["cycle_id"].between(cfg.visible_label_cycle_min, cfg.visible_label_cycle_max)
    frame["fit_allowed"] = frame["split"].eq("train")
    frame["early_stop_allowed"] = frame["split"].eq("val")
    frame["final_eval_only"] = frame["split"].isin(["test", "partial"])
    if targets is not None and not targets.empty:
        keep_cols = [c for c in ["cycle_id", "Q_obs_Ah", "Q_obs_mAh", "SOH_obs", "complete_cycle"] if c in targets.columns]
        frame = frame.merge(targets[keep_cols], on="cycle_id", how="left")
        if "complete_cycle" not in frame.columns:
            frame["complete_cycle"] = ~frame["is_partial"]
        frame.loc[frame["is_partial"], "complete_cycle"] = False
    else:
        frame["complete_cycle"] = ~frame["is_partial"]
    return frame


def make_strict30_manifest(
    capacity_target_csv: Optional[PathLike] = None,
    *,
    cfg: Strict30SplitConfig = Strict30SplitConfig(),
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a JSON-serializable manifest and validate it."""
    targets = None
    cap_hash = ""
    if capacity_target_csv is not None:
        cap_path = Path(capacity_target_csv)
        cap_hash = sha256_file(cap_path)
        targets = load_capacity_targets_strict30(cap_path, cycle_from=cfg.complete_cycle_min, cycle_to=max(cfg.complete_cycle_max, *cfg.partial_cycles))
    split_frame = build_split_frame(targets, cfg)
    counts = {str(k): int(v) for k, v in split_frame["split"].value_counts().sort_index().to_dict().items()}
    manifest: Dict[str, Any] = {
        "model_id": cfg.model_id,
        "protocol": "ASSB111_strict30_visible_test70_heldout",
        "complete_cycle_min": cfg.complete_cycle_min,
        "complete_cycle_max": cfg.complete_cycle_max,
        "visible_label_cycles": list(cfg.visible_label_cycles),
        "train_cycles": list(cfg.train_cycles),
        "val_cycles": list(cfg.val_cycles),
        "test_cycles": list(cfg.test_cycles),
        "partial_cycles": list(cfg.partial_cycles),
        "fit_splits": ["train"],
        "early_stop_splits": ["val"],
        "forbidden_fit_splits": list(cfg.forbidden_fit_splits),
        "split_counts": counts,
        "capacity_target_csv": "" if capacity_target_csv is None else str(capacity_target_csv),
        "capacity_target_sha256": cap_hash,
        "notes": [
            "Only train split may contribute to supervised SOH loss.",
            "Validation split is inside the visible 30% and may be used for early stopping only.",
            "Test split is held out and must not enter loss, scaler fitting, or model selection.",
            "Cycle 522 is partial/report-only for the main complete-cycle metrics.",
        ],
    }
    if extra:
        manifest.update(dict(extra))
    validate_split_manifest(manifest)
    manifest["manifest_sha256_without_self"] = sha256_jsonable({k: v for k, v in manifest.items() if k != "manifest_sha256_without_self"})
    return manifest


def manifest_to_frame(manifest: Mapping[str, Any], targets: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    cfg = Strict30SplitConfig(
        model_id=int(manifest.get("model_id", 111)),
        complete_cycle_min=int(manifest["complete_cycle_min"]),
        complete_cycle_max=int(manifest["complete_cycle_max"]),
        visible_label_cycle_min=int(manifest["visible_label_cycles"][0]),
        visible_label_cycle_max=int(manifest["visible_label_cycles"][1]),
        train_cycle_min=int(manifest["train_cycles"][0]),
        train_cycle_max=int(manifest["train_cycles"][1]),
        val_cycle_min=int(manifest["val_cycles"][0]),
        val_cycle_max=int(manifest["val_cycles"][1]),
        test_cycle_min=int(manifest["test_cycles"][0]),
        test_cycle_max=int(manifest["test_cycles"][1]),
        partial_cycles=tuple(int(x) for x in manifest.get("partial_cycles", [])),
    )
    return build_split_frame(targets, cfg)


def validate_split_manifest(manifest: Mapping[str, Any]) -> None:
    """Hard validation for the no-leakage split."""
    required = [
        "visible_label_cycles", "train_cycles", "val_cycles", "test_cycles",
        "partial_cycles", "forbidden_fit_splits",
    ]
    missing = [k for k in required if k not in manifest]
    if missing:
        raise KeyError(f"split manifest missing required keys: {missing}")
    train = _range_set(manifest["train_cycles"])
    val = _range_set(manifest["val_cycles"])
    test = _range_set(manifest["test_cycles"])
    partial = _to_int_set(manifest.get("partial_cycles", []))
    visible = _range_set(manifest["visible_label_cycles"])
    if train & val or train & test or val & test:
        raise ValueError("train/val/test split ranges overlap")
    if test & partial or train & partial or val & partial:
        raise ValueError("partial cycles overlap with train/val/test")
    if not train <= visible or not val <= visible:
        raise ValueError("train and val cycles must be inside visible_label_cycles")
    if test & visible:
        raise ValueError("test cycles overlap visible-label cycles")
    if min(test) < 160:
        raise ValueError("ASSB111 strict protocol requires test cycles to start at cycle >= 160")
    forbidden = {str(s).lower() for s in manifest.get("forbidden_fit_splits", [])}
    if "test" not in forbidden:
        raise ValueError("forbidden_fit_splits must include test")
    if "partial" not in forbidden:
        raise ValueError("forbidden_fit_splits must include partial")


def write_manifest(manifest: Mapping[str, Any], output_json: PathLike, output_csv: Optional[PathLike] = None) -> None:
    validate_split_manifest(manifest)
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(dict(manifest)), f, ensure_ascii=False, indent=2, sort_keys=True)
    if output_csv is not None:
        frame = manifest_to_frame(manifest)
        out_csv = Path(output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_csv, index=False, encoding="utf-8-sig")


def load_manifest(path: PathLike, *, validate: bool = True) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"split manifest not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if validate:
        validate_split_manifest(manifest)
    return manifest


def split_for_cycles(cycle_ids: Sequence[int], manifest: Mapping[str, Any]) -> np.ndarray:
    """Vectorized split assignment based on a manifest."""
    validate_split_manifest(manifest)
    cycles = np.asarray(cycle_ids, dtype=int)
    out = np.full(cycles.shape, "out_of_scope", dtype=object)
    ranges = {
        "train": manifest["train_cycles"],
        "val": manifest["val_cycles"],
        "test": manifest["test_cycles"],
    }
    for name, pair in ranges.items():
        out[(cycles >= int(pair[0])) & (cycles <= int(pair[1]))] = name
    for c in manifest.get("partial_cycles", []):
        out[cycles == int(c)] = "partial"
    return out


def mask_for_split(cycle_ids: Sequence[int], manifest: Mapping[str, Any], split: Union[str, Sequence[str]]) -> np.ndarray:
    wanted = {split.lower()} if isinstance(split, str) else {str(s).lower() for s in split}
    assigned = np.char.lower(split_for_cycles(cycle_ids, manifest).astype(str))
    mask = np.zeros_like(assigned, dtype=bool)
    for s in wanted:
        mask |= assigned == s
    return mask


def assert_fit_splits_safe(fit_splits: Sequence[str], manifest: Mapping[str, Any]) -> None:
    forbidden = {str(s).strip().lower() for s in manifest.get("forbidden_fit_splits", [])}
    requested = {str(s).strip().lower() for s in fit_splits}
    if "all" in requested:
        raise ValueError("fit_splits='all' is forbidden in ASSB111 strict30")
    bad = sorted(requested & forbidden)
    if bad:
        raise ValueError(f"Forbidden split(s) in fit_splits: {bad}; allowed supervised fit split is train")
    if requested and requested != {"train"}:
        raise ValueError(f"ASSB111 strict30 only allows fit_splits=['train'], got {sorted(requested)}")


__all__ = [
    "Strict30SplitConfig",
    "load_capacity_targets_strict30",
    "assign_split",
    "build_split_frame",
    "make_strict30_manifest",
    "manifest_to_frame",
    "validate_split_manifest",
    "write_manifest",
    "load_manifest",
    "split_for_cycles",
    "mask_for_split",
    "assert_fit_splits_safe",
    "sha256_file",
    "sha256_jsonable",
]
