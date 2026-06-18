# -*- coding: utf-8 -*-
"""
D17-P2 configuration helpers.

The loader intentionally treats configuration files as protocol documents: it
keeps state-supervised/soft-label losses disabled and exposes a small set of
safe defaults for the first 1-profile smoke.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping


DEFAULT_P2_CONFIG: Dict[str, Any] = {
    "d17_protocol_version": 2,
    "experiment_name": "d17_p2_forward_backward_smoke",
    "mode": "voltage_informed_inverse_pinn",
    "seed": 20260615,
    "paths": {
        "split_manifest": "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json",
        "resolved_spec": "configs/resolved_p2dlite_spec_placeholder.json",
        "output_root": "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild",
    },
    "train": {
        "split": "train",
        "profile_index": 0,
        "time_window_s": 40000.0,
        "max_time_points": 4096,
        "n_r": 17,
        "epochs": 100,
        "lr": 1.0e-3,
        "weight_decay": 0.0,
        "device": "auto",
        "gradient_clip_norm": 10.0,
        "allow_softlabel_npz_profile_source": False,
    },
    "model": {
        "hidden_dim": 64,
        "latent_hidden_dim": 64,
        "delta_layers": 3,
        "delta_amp_fraction": 0.018,
        "enable_low_transition_residual": False,
        "low_residual_amp_V": 0.030,
        "use_observed_voltage_in_encoder": True,
        "use_observed_voltage_for_gate": True,
    },
    "loss_weights": {
        "voltage": 1.0,
        "diffusion_pde": 2.0e-6,
        "surface_flux": 2.0e-3,
        "cbar_inventory": 1.0e-3,
        "zero_mean_delta": 1.0,
        "ocp_bv_closure": 1.0e-4,
        "gauge_smooth": 1.0e-4,
        "prior_z": 1.0e-3,
        "residual_preservation": 1.0e-3,
    },
    "checkpoint_selection": {
        "use_voltage_metrics": True,
        "use_physics_metrics": True,
        "use_no_drift_metrics": True,
        "use_state_softlabel_metrics": False,
        "use_frozen_test_metrics": False,
    },
    "audit": {
        "no_state_label_training": True,
        "split_manifest_locked": True,
        "no_oracle_theta0": True,
        "no_test_feedback": True,
        "battery8_policy": "flagged_probe",
    },
}


def _strip_inline_comment(line: str) -> str:
    """Remove YAML comments while respecting simple quoted strings."""
    in_single = False
    in_double = False
    escaped = False
    out = []
    for ch in line:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\" and in_double:
            out.append(ch)
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            out.append(ch)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            out.append(ch)
            continue
        if ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out).rstrip()


def _parse_simple_yaml_scalar(value: str) -> Any:
    """Parse the small YAML subset used by D17 configs without PyYAML.

    Supported: nested mappings by indentation, strings, quoted strings, bools,
    null, integers, floats, and simple inline lists. This intentionally does not
    implement full YAML; it is only a no-dependency fallback for smoke configs.
    """
    v = value.strip()
    if v == "":
        return {}
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    low = v.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none", "~"}:
        return None
    if v.startswith("[") and v.endswith("]"):
        body = v[1:-1].strip()
        if not body:
            return []
        return [_parse_simple_yaml_scalar(part.strip()) for part in body.split(",")]
    try:
        if any(c in v for c in [".", "e", "E"]):
            return float(v)
        return int(v)
    except ValueError:
        return v


def _simple_yaml_load(text: str) -> Dict[str, Any]:
    """Very small YAML mapping parser used when PyYAML is unavailable."""
    root: Dict[str, Any] = {}
    stack: list[tuple[int, MutableMapping[str, Any]]] = [(-1, root)]
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip_inline_comment(raw)
        if not line.strip():
            continue
        if "\t" in line[: len(line) - len(line.lstrip(" \t"))]:
            raise ValueError(f"tabs are not supported in fallback YAML parser at line {lineno}")
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped.startswith("- "):
            raise ValueError(f"fallback YAML parser does not support block lists at line {lineno}")
        if ":" not in stripped:
            raise ValueError(f"expected key: value at line {lineno}: {raw!r}")
        key, value = stripped.split(":", 1)
        key = key.strip().strip('"').strip("'")
        if not key:
            raise ValueError(f"empty key at line {lineno}")
        while indent <= stack[-1][0] and len(stack) > 1:
            stack.pop()
        parent = stack[-1][1]
        value = value.strip()
        if value == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_simple_yaml_scalar(value)
    return root


def _merge_dict(base: MutableMapping[str, Any], update: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for k, v in update.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), MutableMapping):
            _merge_dict(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_P2_CONFIG)
    if path is None:
        return cfg
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        loaded = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
            loaded = yaml.safe_load(text) or {}
        except ModuleNotFoundError:
            # User environment may not have PyYAML. D17 smoke configs use only a
            # small YAML subset, so parse them with an internal no-dependency
            # fallback rather than forcing an extra package install.
            loaded = _simple_yaml_load(text)
        except Exception as exc:
            # If PyYAML is installed but rejects the file, try the same fallback
            # once. This keeps smoke runs robust while still failing clearly for
            # unsupported YAML syntax.
            try:
                loaded = _simple_yaml_load(text)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Could not parse config {path}. PyYAML error: {exc}; "
                    f"fallback parser error: {fallback_exc}"
                ) from fallback_exc
    if not isinstance(loaded, dict):
        raise ValueError("config file must define a JSON/YAML object")
    _merge_dict(cfg, loaded)
    enforce_no_state_label_config(cfg)
    return cfg


def enforce_no_state_label_config(cfg: Mapping[str, Any]) -> None:
    """Fail fast if the P2 config tries to enable state-label training."""
    losses = cfg.get("losses", {}) if isinstance(cfg.get("losses", {}), Mapping) else {}
    forbidden_loss_flags = [
        "state_supervised",
        "softlabel_supervised",
        "cs_soft",
        "theta_soft",
        "phie_soft",
        "phis_c_soft",
        "cs_a_soft",
        "cs_c_soft",
        "theta_a_soft",
        "theta_c_soft",
    ]
    bad = [k for k in forbidden_loss_flags if bool(losses.get(k, False))]
    if bad:
        raise ValueError(f"D17-P2 forbids state/soft-label supervised losses; enabled keys: {bad}")
    ckpt = cfg.get("checkpoint_selection", {}) if isinstance(cfg.get("checkpoint_selection", {}), Mapping) else {}
    if bool(ckpt.get("use_state_softlabel_metrics", False)) or bool(ckpt.get("use_frozen_test_metrics", False)):
        raise ValueError("D17-P2 checkpoint selection cannot use state soft-label or frozen-test metrics")


def cfg_get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur
