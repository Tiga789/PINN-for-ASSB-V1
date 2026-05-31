"""D12 opt-in metadata runtime for GV1 metadata on/off ablation.

This module is additive: it does not edit D9.6/D9.5.1 source files.  The
D12 wrapper imports this module, registers a process-local monkey patch, and
then delegates to the existing D9.6/D9.5.1 trainer.  The patch appends an
optional profile-level metadata vector to ``GV1ReplayDataset.condition`` after
that dataset has been built by the original trainer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


PROFILE_COLS = ("profile_id", "cell_uid", "profile_key", "label", "run_id")
PATH_COLS = ("profile_npz", "solution_npz", "npz_path", "prepared_dir", "profile_path", "source_file")


@dataclass
class D12MetadataRuntimeConfig:
    mode: str = "off"  # off | zero | on
    metadata_manifest: str | None = None
    profile_id: str | None = None
    feature_columns: str | None = "auto"
    strict_profile_match: bool = True
    allow_target_probe: bool = False
    target_profile_id: str = "Batch-1_2C_battery-8"
    runtime_tag: str = "d12_metadata_runtime"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class D12MetadataVector:
    ok: bool
    mode: str
    profile_id: str | None
    matched_by: str | None
    metadata_dim: int
    feature_names: list[str]
    feature_vector: list[float]
    warnings: list[str]
    row_preview: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CONFIG = D12MetadataRuntimeConfig()
_PATCHED = False
_ORIGINAL_INIT = None


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def _bool_or_float(x: Any) -> float | None:
    text = str(x or "").strip().lower()
    if text in {"true", "yes", "y", "on", "pass"}:
        return 1.0
    if text in {"false", "no", "n", "off", "fail", ""}:
        return 0.0
    try:
        val = float(text)
    except Exception:
        return None
    return val if math.isfinite(val) else None


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", newline="", encoding="utf-8-sig") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _infer_profile_id_from_npz(solution_npz: str | Path | None) -> str | None:
    if not solution_npz:
        return None
    p = Path(solution_npz)
    if p.exists():
        try:
            with np.load(p, allow_pickle=True) as z:
                for key in PROFILE_COLS:
                    if key in z.files:
                        a = np.asarray(z[key]).reshape(-1)
                        if a.size:
                            v = a[0]
                            if isinstance(v, bytes):
                                v = v.decode("utf-8", "ignore")
                            if str(v).strip():
                                return str(v).strip()
                parts = {}
                for key in ("batch_id", "protocol", "battery_id"):
                    if key in z.files:
                        a = np.asarray(z[key]).reshape(-1)
                        if a.size:
                            v = a[0]
                            if isinstance(v, bytes):
                                v = v.decode("utf-8", "ignore")
                            parts[key] = str(v).strip()
                if len(parts) == 3:
                    return f"{parts['batch_id']}_{parts['protocol']}_{parts['battery_id']}"
        except Exception:
            pass
    # Common profile directory fallback.
    for cand in (p.parent.name, p.stem):
        if cand and "battery" in cand.lower():
            return cand
    return None


def _row_profile_id(row: Mapping[str, Any]) -> str | None:
    for c in PROFILE_COLS:
        if str(row.get(c, "")).strip():
            return str(row[c]).strip()
    bits = [str(row.get(c, "")).strip() for c in ("batch_id", "protocol", "battery_id")]
    return "_".join(bits) if all(bits) else None


def _match_row(rows: Sequence[Mapping[str, str]], profile_id: str | None, solution_npz: str | Path | None) -> tuple[dict[str, str] | None, str | None]:
    if profile_id:
        npid = _norm(profile_id)
        for r in rows:
            for c in PROFILE_COLS:
                if c in r and _norm(r.get(c)) == npid:
                    return dict(r), f"{c}=profile_id"
            comp = "_".join(str(r.get(c, "")).strip() for c in ("batch_id", "protocol", "battery_id"))
            if _norm(comp) == npid:
                return dict(r), "batch_protocol_battery=profile_id"
    if solution_npz:
        sp = str(Path(solution_npz)).replace("\\", "/").lower()
        sn = Path(solution_npz).name.lower()
        for r in rows:
            for c in PATH_COLS:
                val = str(r.get(c, "")).replace("\\", "/").lower()
                if val and (val == sp or sn in val):
                    return dict(r), f"{c}=solution_path"
    return None, None


def _auto_feature_columns(rows: Sequence[Mapping[str, str]]) -> list[str]:
    if not rows:
        return []
    banned = ("mae", "rmse", "corr", "r2", "target_value", "peer_median", "robust_z", "abs_z")
    out = []
    for col in rows[0].keys():
        low = col.lower()
        if any(b in low for b in banned):
            continue
        if low.startswith("d11c2_") or low.startswith("d12_"):
            vals = [_bool_or_float(r.get(col)) for r in rows[:50]]
            if any(v is not None for v in vals):
                out.append(col)
    return out


def _generated_features(row: Mapping[str, Any]) -> tuple[list[str], list[float]]:
    batch = str(row.get("batch_id", "")).strip().lower()
    protocol = str(row.get("protocol", "")).strip().lower()
    battery = str(row.get("battery_id", "")).strip().lower()
    names, vals = [], []
    def add(n: str, v: float) -> None:
        names.append(n); vals.append(float(v))
    flag = _bool_or_float(row.get("d11c2_is_b1_2c_battery8", row.get("is_b1_2c_battery8", 0)))
    add("d12meta_is_b1_2c_battery8", flag or 0.0)
    for b in ("batch-1", "batch-3", "batch-4"):
        add("d12meta_" + b.replace("-", ""), 1.0 if batch == b else 0.0)
    for p in ("2c", "r2.5", "r3"):
        add("d12meta_protocol_" + p.replace(".", "p"), 1.0 if protocol == p else 0.0)
    m = re.search(r"(\d+)", battery)
    add("d12meta_battery_index_over8", (float(m.group(1)) / 8.0) if m else 0.0)
    scope = str(row.get("d11c2_training_scope", row.get("training_scope", ""))).lower()
    add("d12meta_scope_excluded", 1.0 if "excluded" in scope else 0.0)
    return names, vals


def build_metadata_vector(cfg: D12MetadataRuntimeConfig, solution_npz: str | Path | None) -> D12MetadataVector:
    mode = (cfg.mode or "off").lower().strip()
    if mode not in {"off", "zero", "on"}:
        raise ValueError("D12 metadata mode must be off, zero, or on")
    if mode == "off":
        return D12MetadataVector(True, mode, None, None, 0, [], [], [], {})
    if not cfg.metadata_manifest:
        raise ValueError("metadata_manifest is required for zero/on mode")
    rows = _read_csv(cfg.metadata_manifest)
    profile_id = cfg.profile_id or _infer_profile_id_from_npz(solution_npz)
    row, matched_by = _match_row(rows, profile_id, solution_npz)
    warnings = []
    if row is None:
        if cfg.strict_profile_match:
            raise RuntimeError(f"Could not match profile {profile_id!r} in {cfg.metadata_manifest}")
        row = rows[0] if rows else {}
        matched_by = "unmatched_non_strict_first_schema_zero"
        warnings.append("No matching metadata row; using zero vector from first-row schema.")
    matched_profile = _row_profile_id(row) or profile_id
    if (not cfg.allow_target_probe) and _norm(matched_profile) == _norm(cfg.target_profile_id):
        raise RuntimeError("Matched battery-8 target probe; refusing unless metadata_allow_target_probe is true.")
    explicit = None if cfg.feature_columns in {None, "", "auto"} else [x.strip() for x in str(cfg.feature_columns).split(",") if x.strip()]
    cols = explicit if explicit is not None else _auto_feature_columns(rows)
    names, vals = [], []
    for col in cols:
        names.append(col)
        vals.append(_bool_or_float(row.get(col)) or 0.0)
    gnames, gvals = _generated_features(row)
    for n, v in zip(gnames, gvals):
        if n not in names:
            names.append(n); vals.append(v)
    if mode == "zero":
        vals = [0.0 for _ in vals]
    return D12MetadataVector(True, mode, matched_profile, matched_by, len(vals), names, [float(v) for v in vals], warnings, {k: row.get(k) for k in list(row)[:25]})


def configure_runtime(cfg: D12MetadataRuntimeConfig | Mapping[str, Any]) -> None:
    global _CONFIG
    _CONFIG = cfg if isinstance(cfg, D12MetadataRuntimeConfig) else D12MetadataRuntimeConfig(**dict(cfg))


def register_patch() -> None:
    global _PATCHED, _ORIGINAL_INIT
    if _PATCHED:
        return
    import gv1.trainer as tr
    cls = tr.GV1ReplayDataset
    _ORIGINAL_INIT = cls.__init__
    def patched_init(self: Any, arrays: Mapping[str, np.ndarray], cfg: Any) -> None:
        _ORIGINAL_INIT(self, arrays, cfg)
        result = build_metadata_vector(_CONFIG, getattr(cfg, "solution_npz", None))
        if result.metadata_dim:
            self.condition = np.concatenate([np.asarray(self.condition, dtype=np.float32).reshape(-1), np.asarray(result.feature_vector, dtype=np.float32)], axis=0)
        self.d12_metadata_runtime = result.to_dict()
        try:
            out = Path(getattr(cfg, "output_dir"))
            out.mkdir(parents=True, exist_ok=True)
            (out / "d12_metadata_runtime_summary.json").write_text(json.dumps({"ok": True, "stage": "D12 runtime metadata patch", "metadata": result.to_dict(), "runtime_config": _CONFIG.to_dict()}, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    cls.__init__ = patched_init
    _PATCHED = True


__all__ = ["D12MetadataRuntimeConfig", "D12MetadataVector", "configure_runtime", "register_patch", "build_metadata_vector"]
