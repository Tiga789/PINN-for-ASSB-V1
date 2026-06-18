# -*- coding: utf-8 -*-
"""Small D17 audit helpers shared by future stages."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def stable_json_hash(obj: Dict[str, Any]) -> str:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def promotion_gate_template() -> Dict[str, Any]:
    return {
        "no_state_label_training": False,
        "split_manifest_locked": False,
        "no_oracle_theta0": False,
        "no_test_feedback": False,
        "cbar_hard_conservation": False,
        "zero_mean_delta_pass": False,
        "local_residual_no_leak": False,
    }
