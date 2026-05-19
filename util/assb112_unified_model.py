# -*- coding: utf-8 -*-
"""ModelFin_112A L1 unified five-target model wrapper.

L1 is an engineering single-model interface: one model directory, one selected
checkpoint, one configuration, and one forward/evaluator path.  The four state
variables are protected by a frozen ModelFin_107A state source; SOH is predicted
by the robust strict30 SOH head.  This module does not pretend that SOH already
couples back into the electrochemical state core.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union
import json
import math

import numpy as np
import pandas as pd

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from util.assb111_soh_model import Assb111SOHHead
try:
    from util.assb_soh_feature_schema import transform_with_scaler
except Exception:  # pragma: no cover
    from util.assb111_feature_schema import transform_with_scaler  # type: ignore

PathLike = Union[str, Path]


@dataclass
class ASSB112UnifiedConfig:
    model_name: str = "ModelFin_112A_unified5_frozen107A_robustSOH"
    state_core_type: str = "frozen_107A_eval_npz"
    state_core_dir: str = "ModelFin_107A"
    state_eval_npz: str = "EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only/evaluation_paired.npz"
    soh_head_dir: str = "ModelFin_112_robustSOH_seed42"
    feature_schema_json: str = "feature_schema.json"
    split_manifest_json: str = "split_manifest.json"
    notes: str = "L1 engineering unified interface; frozen state core + robust SOH head."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ASSB112UnifiedConfig":
        return cls(**{k: v for k, v in dict(d).items() if k in cls.__dataclass_fields__})


def _json_clean(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_json_clean(v) for v in x.tolist()]
    if torch is not None and torch.is_tensor(x):
        return _json_clean(x.detach().cpu().numpy())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        val = float(x)
        return None if not math.isfinite(val) else val
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    return x


def save_json(obj: Mapping[str, Any], path: PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(dict(obj)), f, ensure_ascii=False, indent=2, sort_keys=True)


class ASSB112UnifiedModel:
    def __init__(self, cfg: ASSB112UnifiedConfig, *, model_dir: Optional[PathLike] = None, map_location: str = "cpu"):
        self.cfg = cfg
        self.model_dir = Path(model_dir) if model_dir is not None else None
        soh_dir = self._resolve_path(cfg.soh_head_dir)
        self.soh_head = Assb111SOHHead.load(soh_dir, map_location=map_location)
        cfg_path = soh_dir / "soh_head_config.json"
        with cfg_path.open("r", encoding="utf-8") as f:
            self.soh_payload = json.load(f)
        self.feature_columns = list(self.soh_payload.get("feature_columns") or self.soh_payload.get("scaler", {}).get("feature_columns", []))
        self.scaler = self.soh_payload.get("scaler", None)
        self.state_npz_path = self._resolve_path(cfg.state_eval_npz)
        self._state_npz_cache: Optional[Mapping[str, np.ndarray]] = None

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        if self.model_dir is not None and (self.model_dir / path).exists():
            return self.model_dir / path
        return path

    @classmethod
    def load(cls, model_dir: PathLike, *, map_location: str = "cpu") -> "ASSB112UnifiedModel":
        md = Path(model_dir)
        cfg_path = md / "unified_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(cfg_path)
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = ASSB112UnifiedConfig.from_dict(json.load(f))
        return cls(cfg, model_dir=md, map_location=map_location)

    def load_state_npz(self) -> Mapping[str, np.ndarray]:
        if self._state_npz_cache is None:
            if not self.state_npz_path.exists():
                raise FileNotFoundError(self.state_npz_path)
            data = np.load(self.state_npz_path, allow_pickle=True)
            self._state_npz_cache = {k: data[k] for k in data.files}
        return self._state_npz_cache

    @staticmethod
    def _find_key(data: Mapping[str, np.ndarray], candidates: Sequence[str]) -> Optional[str]:
        keys_lower = {str(k).lower(): k for k in data.keys()}
        for cand in candidates:
            if cand.lower() in keys_lower:
                return keys_lower[cand.lower()]
        for cand in candidates:
            low = cand.lower()
            for k in data.keys():
                if low in str(k).lower():
                    return k
        return None

    def predict_soh(self, feature_frame: pd.DataFrame, *, device: str = "cpu") -> pd.DataFrame:
        if self.scaler is None:
            raise RuntimeError("SOH head payload has no scaler; cannot transform feature_frame")
        x_np = transform_with_scaler(feature_frame, self.scaler)
        cycles = feature_frame["cycle_id"].astype(int).to_numpy() if "cycle_id" in feature_frame.columns else np.arange(len(feature_frame))
        delta_np = np.ones_like(cycles, dtype=float)
        if len(delta_np) > 1:
            delta_np[1:] = np.maximum(1.0, np.diff(cycles).astype(float))
        if torch is None:  # pragma: no cover
            raise RuntimeError(f"PyTorch required: {_TORCH_IMPORT_ERROR}")
        self.soh_head.to(device=device)
        self.soh_head.eval()
        with torch.no_grad():
            x = torch.as_tensor(x_np, dtype=torch.float64 if str(self.soh_head.cfg.dtype).lower().startswith("float64") else torch.float32, device=device)
            delta = torch.as_tensor(delta_np, dtype=x.dtype, device=device)
            out = self.soh_head(x, delta_cycle=delta)
        pred = out.SOH_pred.detach().cpu().numpy().reshape(-1)
        res = pd.DataFrame({"cycle_id": cycles.astype(int), "SOH_pred": pred})
        if "split" in feature_frame.columns:
            res["split"] = feature_frame["split"].astype(str).to_numpy()
        if "SOH_obs" in feature_frame.columns:
            res["SOH_obs"] = pd.to_numeric(feature_frame["SOH_obs"], errors="coerce").to_numpy(dtype=float)
        return res

    def get_state_arrays(self) -> Dict[str, np.ndarray]:
        """Return predicted/true state arrays from the frozen state NPZ.

        The evaluator uses this method with key heuristics, so it can work with
        several historical EvalFin_107A NPZ naming conventions.
        """
        data = self.load_state_npz()
        candidate_map = {
            "cs_a_pred": ["cs_a_pred", "pred_cs_a", "cs_a_hat", "cs_a_corrected_pred"],
            "cs_a_true": ["cs_a_true", "true_cs_a", "cs_a_label"],
            "cs_c_pred": ["cs_c_pred", "pred_cs_c", "cs_c_hat"],
            "cs_c_true": ["cs_c_true", "true_cs_c", "cs_c_label"],
            "phie_pred": ["phie_pred", "pred_phie", "phie_hat"],
            "phie_true": ["phie_true", "true_phie", "phie_label"],
            "phis_c_pred": ["phis_c_pred", "pred_phis_c", "phis_c_hat"],
            "phis_c_true": ["phis_c_true", "true_phis_c", "phis_c_label"],
            "cycle_id": ["cycle_id", "cycles", "cycle_ids"],
        }
        out: Dict[str, np.ndarray] = {}
        missing: Dict[str, Sequence[str]] = {}
        for name, cands in candidate_map.items():
            key = self._find_key(data, cands)
            if key is not None:
                out[name] = np.asarray(data[key])
            elif name != "cycle_id":
                missing[name] = cands
        if missing:
            out["_missing_keys"] = np.asarray([json.dumps(missing, ensure_ascii=False)], dtype=object)
        return out

    def forward(self, feature_frame: pd.DataFrame, *, device: str = "cpu") -> Dict[str, Any]:
        return {"soh": self.predict_soh(feature_frame, device=device), "states": self.get_state_arrays(), "config": self.cfg.to_dict()}


__all__ = ["ASSB112UnifiedConfig", "ASSB112UnifiedModel", "save_json"]
