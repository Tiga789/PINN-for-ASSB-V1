# -*- coding: utf-8 -*-
"""
D17-P1 resolved spec loader and range audit.

The spec is a prior/metadata source. It must not contain per-profile state answers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


RECOMMENDED_TOP_LEVEL_KEYS = (
    "cell",
    "chemistry",
    "capacity",
    "geometry",
    "ocp",
    "transport",
    "kinetics",
    "temperature",
)

SUSPICIOUS_STATE_KEYS = (
    "cs_a",
    "cs_c",
    "theta_a",
    "theta_c",
    "phie",
    "phis_c",
    "theta0_oracle",
    "oracle_shift",
)


def load_resolved_spec(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"resolved spec not found: {path}")
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("resolved spec must be a JSON object")
    return spec


def _walk_keys(obj: Any, prefix: str = "") -> List[str]:
    keys: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            keys.append(p)
            keys.extend(_walk_keys(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:10]):
            keys.extend(_walk_keys(v, f"{prefix}[{i}]"))
    return keys


def audit_resolved_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    keys = _walk_keys(spec)
    missing_recommended = [k for k in RECOMMENDED_TOP_LEVEL_KEYS if k not in spec]
    suspicious = [k for k in keys if any(tok == k.split(".")[-1] for tok in SUSPICIOUS_STATE_KEYS)]
    report = {
        "pass": len(suspicious) == 0,
        "missing_recommended_top_level_keys": missing_recommended,
        "suspicious_state_answer_keys": suspicious,
        "notes": [
            "A missing recommended key is REVIEW, not FAIL, because D17-P1 only audits data protocol.",
            "Any cs/theta/phie/phis or theta0_oracle key in resolved spec is suspicious for D17 mainline.",
        ],
    }
    return report
