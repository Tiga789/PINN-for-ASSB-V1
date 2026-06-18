# -*- coding: utf-8 -*-
"""
D17-P1 observed-only dataset utilities.

Training data may include only observed fields such as I(t), V(t), T(t), time,
cycle/step/protocol metadata, and priors.  State soft-label arrays are not
loaded here.

Important:
- `softlabel_npz` paths in the split manifest are for later frozen audit only.
- By default this module refuses to fall back to softlabel NPZ files as training
  profile sources, even if they contain observed fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


ALLOWED_PROFILE_FIELDS = (
    "t_global_s",
    "time_s",
    "I_profile",
    "current_A",
    "voltage_exp",
    "temperature_C",
    "cycle_id",
    "step_id",
    "step_type",
    "protocol",
    "batch",
    "cell_uid",
    "source_file",
)

FORBIDDEN_STATE_FIELDS = (
    "cs_a",
    "cs_c",
    "theta_a",
    "theta_c",
    "phie",
    "phis_c",
    "phie_soft",
    "phis_c_soft",
    "cs_a_soft",
    "cs_c_soft",
    "theta_a_soft",
    "theta_c_soft",
)


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.dtype.kind in ("U", "S", "O") and value.size == 1:
            return value.reshape(-1)[0].item() if hasattr(value.reshape(-1)[0], "item") else str(value.reshape(-1)[0])
    return value


def load_observed_profile(
    replay_npz: str | Path,
    allowed_fields: Iterable[str] = ALLOWED_PROFILE_FIELDS,
    max_time_points: Optional[int] = None,
) -> Dict[str, Any]:
    """Load observed fields from a replay profile NPZ.

    This function does not read state soft-label arrays. If forbidden state keys
    are present in the file, they are ignored and reported under `_ignored_state_keys`.
    """
    replay_npz = Path(replay_npz)
    if not replay_npz.exists():
        raise FileNotFoundError(f"Replay profile not found: {replay_npz}")

    allowed = set(allowed_fields)
    out: Dict[str, Any] = {"_source_npz": str(replay_npz)}
    ignored_state_keys: List[str] = []
    with np.load(replay_npz, allow_pickle=True) as data:
        keys = set(data.files)
        for k in FORBIDDEN_STATE_FIELDS:
            if k in keys:
                ignored_state_keys.append(k)
        for k in allowed:
            if k in keys:
                value = data[k]
                value = _safe_scalar(value)
                if isinstance(value, np.ndarray) and max_time_points is not None and value.ndim >= 1 and value.shape[0] > max_time_points:
                    # Deterministic downsample to avoid gigantic smoke tensors.
                    idx = np.linspace(0, value.shape[0] - 1, max_time_points).round().astype(int)
                    value = value[idx]
                out[k] = value
    out["_ignored_state_keys"] = ignored_state_keys
    required_any_time = "t_global_s" in out or "time_s" in out
    required_any_current = "I_profile" in out or "current_A" in out
    if not required_any_time or not required_any_current or "voltage_exp" not in out:
        raise ValueError(
            f"Observed profile is missing required observed fields. "
            f"Need time, current, voltage_exp. Found keys: {sorted(out.keys())}"
        )
    return out


class D17ProfileDataset:
    """Manifest-backed observed-only profile list.

    This class is deliberately minimal for P1. Full torch Dataset behavior should
    be added in D17-P2/P3 after the no-label audit passes.
    """

    def __init__(
        self,
        split_manifest: str | Path,
        split: str = "train",
        allow_softlabel_npz_profile_source: bool = False,
    ) -> None:
        self.split_manifest = Path(split_manifest)
        self.manifest = json.loads(self.split_manifest.read_text(encoding="utf-8"))
        self.split = split
        self.allow_softlabel_npz_profile_source = allow_softlabel_npz_profile_source
        self.records: List[Dict[str, Any]] = [
            r for r in self.manifest.get("records", []) if r.get("split") == split
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        replay_npz = rec.get("replay_npz")
        if not replay_npz:
            if self.allow_softlabel_npz_profile_source:
                # This option exists only for emergency smoke tests. It is off by default
                # and should not be used for formal no-state-label training.
                replay_npz = rec.get("softlabel_npz")
            else:
                raise RuntimeError(
                    f"No replay_npz for {rec.get('canonical_cell_uid')}. "
                    "Do not use softlabel_npz as profile source in formal D17 training."
                )
        profile = load_observed_profile(replay_npz)
        profile["_manifest_record"] = rec
        return profile

    def summary(self) -> Dict[str, Any]:
        missing_replay = [r.get("canonical_cell_uid") for r in self.records if not r.get("replay_npz")]
        return {
            "split": self.split,
            "count": len(self.records),
            "missing_replay_count": len(missing_replay),
            "missing_replay": missing_replay[:100],
            "manifest_hash_sha256": self.manifest.get("manifest_hash_sha256"),
            "no_state_label_policy": "observed-only; state soft-label arrays are ignored/refused for training",
        }
