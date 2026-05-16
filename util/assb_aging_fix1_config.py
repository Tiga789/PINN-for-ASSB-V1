# -*- coding: utf-8 -*-
"""Configuration helpers for ASSB aging-fix1 / ModelFin_110.

This module is intentionally standalone.  It centralizes all aging-related
configuration, default values, safety guards, and serialization utilities so the
future Stage-C patches to ``init_pinn.py`` / ``_losses.py`` do not duplicate
constants.

Engineering rules encoded here:
- The original pointwise soft-label ``DATA_LOSS`` must remain disabled.
- ``c_s,max`` is treated as an invariant material/normalization scale.
- Aging is represented by slow cycle-level mechanisms: LAM_c, positive-electrode
  usable theta-window shrinkage, and optional R_ohm growth.
- Stage B trains the low-dimensional mechanism only; Stage C injects that
  mechanism into effective-SPM closure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union
import json
import math

PathLike = Union[str, Path]


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "none", "null", ""}:
        return False
    return bool(default)


def _get(params: Dict[str, Any], names: Iterable[str], default: Any) -> Any:
    for name in names:
        if name in params and params[name] is not None:
            return params[name]
        upper = name.upper()
        if upper in params and params[upper] is not None:
            return params[upper]
    return default


def _get_float(params: Dict[str, Any], names: Iterable[str], default: float) -> float:
    value = _get(params, names, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _get_int(params: Dict[str, Any], names: Iterable[str], default: int) -> int:
    value = _get(params, names, default)
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _get_str(params: Dict[str, Any], names: Iterable[str], default: str) -> str:
    value = _get(params, names, default)
    return str(value)


def _get_bool(params: Dict[str, Any], names: Iterable[str], default: bool) -> bool:
    return _to_bool(_get(params, names, default), default=default)


@dataclass
class AgingFix1Config:
    """Aging mechanism configuration used by Stage B and Stage C.

    The defaults are deliberately conservative but not flat: the initial final
    damage amplitudes are near 50% of the allowed ranges, so Stage B cannot get
    stuck at the 109-style ``SOH_min ~= 0.95`` simply because of initialization.
    """

    # Experiment identity and data paths.
    experiment_id: int = 110
    stage: str = "B_MECHANISM"
    use_assb_aging_fix1: bool = True
    freeze_107a_core: bool = True
    load_model: str = "ModelFin_107A/best.pt"
    load_aging_model: str = "ModelFin_110_stageB/aging_mechanism.pt"
    cycle_table_csv: str = "Data/assb_aging_fix1/cycle_table.csv"
    capacity_target_csv: str = "Data/assb_capacity_soh_targets/capacity_soh_targets.csv"

    # Mechanism degrees of freedom.
    feature_dim: int = 8
    hidden_dim: int = 32
    hidden_layers: int = 2
    lam_max: float = 0.60
    window_loss_max: float = 0.45
    r_ohm_base: float = 105.0
    r_ohm_delta_max: float = 250.0
    enable_lli_shift: bool = False
    lli_shift_max: float = 0.0
    use_apparent_capacity: bool = False
    apparent_gamma_r: float = 0.0

    # Initialization and monotonicity.
    init_amplitude_logit_lam: float = 0.0
    init_amplitude_logit_window: float = 0.0
    init_amplitude_logit_rohm: float = -2.0
    init_rate_bias: float = 0.0
    min_rate: float = 1.0e-5
    eps: float = 1.0e-12

    # Capacity/SOH loss weights.
    huber_delta: float = 0.015
    w_q: float = 1.0
    w_soh: float = 5.0
    w_smooth: float = 0.05
    w_rate: float = 0.01
    w_bounds: float = 0.1
    w_final: float = 10.0
    w_lam_window_balance: float = 0.01

    # Stage-C injection switches.
    use_injection_cbar: bool = False
    use_injection_flux: bool = False
    use_injection_theta_window: bool = False
    use_injection_rohm: bool = False
    lock_common_mode_gauge: bool = True

    # Safety.
    data_loss: bool = False
    alpha_data: float = 0.0
    max_batch_size_data: int = 0
    save_aging_config_json: bool = True
    save_aging_state_pt: bool = True
    dtype: str = "float64"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgingFix1Config":
        base = cls()
        valid = set(base.to_dict())
        kwargs = {k: v for k, v in data.items() if k in valid}
        return cls(**kwargs)

    def normalized_stage(self) -> str:
        return str(self.stage).strip().upper()

    def is_stage_b(self) -> bool:
        return self.normalized_stage().startswith("B") or "MECHANISM" in self.normalized_stage()

    def is_stage_c(self) -> bool:
        return self.normalized_stage().startswith("C") or "INJECT" in self.normalized_stage()


def parse_aging_fix1_config(params: Optional[Dict[str, Any]] = None) -> AgingFix1Config:
    """Parse config from an input-parameter dictionary.

    The parser accepts both concise Python-style names and historical uppercase
    input-file names.  Unknown fields are ignored so this can be called safely
    from older ``init_pinn.py`` versions.
    """
    params = dict(params or {})
    cfg = AgingFix1Config(
        experiment_id=_get_int(params, ["ID", "experiment_id"], 110),
        stage=_get_str(params, ["AGING_STAGE", "stage"], "B_MECHANISM"),
        use_assb_aging_fix1=_get_bool(params, ["USE_ASSB_AGING_FIX1", "use_assb_aging_fix1"], True),
        freeze_107a_core=_get_bool(params, ["FREEZE_107A_CORE", "freeze_107a_core"], True),
        load_model=_get_str(params, ["LOAD_MODEL", "load_model"], "ModelFin_107A/best.pt"),
        load_aging_model=_get_str(params, ["LOAD_AGING_MODEL", "load_aging_model"], "ModelFin_110_stageB/aging_mechanism.pt"),
        cycle_table_csv=_get_str(params, ["AGING_CYCLE_TABLE_CSV", "cycle_table_csv"], "Data/assb_aging_fix1/cycle_table.csv"),
        capacity_target_csv=_get_str(params, ["AGING_CAPACITY_TARGET_CSV", "capacity_target_csv"], "Data/assb_capacity_soh_targets/capacity_soh_targets.csv"),
        feature_dim=_get_int(params, ["AGING_FEATURE_DIM", "feature_dim"], 8),
        hidden_dim=_get_int(params, ["AGING_HIDDEN_DIM", "hidden_dim"], 32),
        hidden_layers=_get_int(params, ["AGING_HIDDEN_LAYERS", "hidden_layers"], 2),
        lam_max=_get_float(params, ["AGING_LAM_MAX", "lam_max"], 0.60),
        window_loss_max=_get_float(params, ["AGING_WINDOW_LOSS_MAX", "window_loss_max"], 0.45),
        r_ohm_base=_get_float(params, ["AGING_R_OHM_BASE", "R_ohm_eff", "r_ohm_base"], 105.0),
        r_ohm_delta_max=_get_float(params, ["AGING_ROHM_DELTA_MAX", "AGING_R_OHM_DELTA_MAX", "r_ohm_delta_max"], 250.0),
        enable_lli_shift=_get_bool(params, ["AGING_ENABLE_LLI_SHIFT", "enable_lli_shift"], False),
        lli_shift_max=_get_float(params, ["AGING_LLI_SHIFT_MAX", "lli_shift_max"], 0.0),
        use_apparent_capacity=_get_bool(params, ["AGING_USE_APPARENT_CAPACITY", "use_apparent_capacity"], False),
        apparent_gamma_r=_get_float(params, ["AGING_APPARENT_GAMMA_R", "apparent_gamma_r"], 0.0),
        init_amplitude_logit_lam=_get_float(params, ["AGING_INIT_AMPLITUDE_LOGIT_LAM", "init_amplitude_logit_lam"], 0.0),
        init_amplitude_logit_window=_get_float(params, ["AGING_INIT_AMPLITUDE_LOGIT_WINDOW", "init_amplitude_logit_window"], 0.0),
        init_amplitude_logit_rohm=_get_float(params, ["AGING_INIT_AMPLITUDE_LOGIT_ROHM", "init_amplitude_logit_rohm"], -2.0),
        init_rate_bias=_get_float(params, ["AGING_INIT_RATE_BIAS", "init_rate_bias"], 0.0),
        min_rate=_get_float(params, ["AGING_MIN_RATE", "min_rate"], 1.0e-5),
        huber_delta=_get_float(params, ["AGING_HUBER_DELTA", "huber_delta"], 0.015),
        w_q=_get_float(params, ["AGING_W_Q", "w_q"], 1.0),
        w_soh=_get_float(params, ["AGING_W_SOH", "w_soh"], 5.0),
        w_smooth=_get_float(params, ["AGING_W_SMOOTH", "w_smooth"], 0.05),
        w_rate=_get_float(params, ["AGING_W_RATE", "w_rate"], 0.01),
        w_bounds=_get_float(params, ["AGING_W_BOUNDS", "w_bounds"], 0.1),
        w_final=_get_float(params, ["AGING_W_FINAL", "w_final"], 10.0),
        w_lam_window_balance=_get_float(params, ["AGING_W_LAM_WINDOW_BALANCE", "w_lam_window_balance"], 0.01),
        use_injection_cbar=_get_bool(params, ["USE_ASSB_AGING_INJECTION_CBAR", "use_injection_cbar"], False),
        use_injection_flux=_get_bool(params, ["USE_ASSB_AGING_INJECTION_FLUX", "use_injection_flux"], False),
        use_injection_theta_window=_get_bool(params, ["USE_ASSB_AGING_INJECTION_THETA_WINDOW", "use_injection_theta_window"], False),
        use_injection_rohm=_get_bool(params, ["USE_ASSB_AGING_INJECTION_ROHM", "use_injection_rohm"], False),
        lock_common_mode_gauge=_get_bool(params, ["LOCK_COMMON_MODE_GAUGE", "lock_common_mode_gauge"], True),
        data_loss=_get_bool(params, ["DATA_LOSS", "data_loss"], False),
        alpha_data=_get_float(params, ["ALPHA_DATA", "alpha_data"], 0.0),
        max_batch_size_data=_get_int(params, ["MAX_BATCH_SIZE_DATA", "max_batch_size_data"], 0),
        save_aging_config_json=_get_bool(params, ["SAVE_AGING_CONFIG_JSON", "save_aging_config_json"], True),
        save_aging_state_pt=_get_bool(params, ["SAVE_AGING_STATE_PT", "save_aging_state_pt"], True),
        dtype=_get_str(params, ["AGING_DTYPE", "dtype"], "float64"),
    )
    return validate_aging_config(cfg)


def validate_aging_config(cfg: AgingFix1Config, *, allow_data_loss: bool = False) -> AgingFix1Config:
    """Validate and return *cfg*.

    Raises a RuntimeError for project-breaking settings.  This guard is meant to
    be called from Stage-C ``init_pinn.py`` as well as standalone scripts.
    """
    errors = []
    if cfg.feature_dim <= 0:
        errors.append("feature_dim must be positive")
    if cfg.hidden_dim <= 0:
        errors.append("hidden_dim must be positive")
    if cfg.hidden_layers < 0:
        errors.append("hidden_layers must be non-negative")
    for name, value in {
        "lam_max": cfg.lam_max,
        "window_loss_max": cfg.window_loss_max,
        "r_ohm_delta_max": cfg.r_ohm_delta_max,
        "huber_delta": cfg.huber_delta,
    }.items():
        if not math.isfinite(float(value)) or float(value) < 0.0:
            errors.append(f"{name} must be finite and non-negative")
    if cfg.lam_max >= 0.95:
        errors.append("lam_max >= 0.95 is not allowed for Stage-B first pass")
    if cfg.window_loss_max >= 0.95:
        errors.append("window_loss_max >= 0.95 is not allowed for Stage-B first pass")
    if (cfg.data_loss or cfg.alpha_data > 0.0 or cfg.max_batch_size_data > 0) and not allow_data_loss:
        errors.append(
            "Original DATA_LOSS must remain disabled: expected DATA_LOSS=False, "
            "ALPHA_DATA=0, MAX_BATCH_SIZE_DATA=0. Use --allow_data_loss only for explicit ablations."
        )
    if cfg.use_injection_rohm and not cfg.lock_common_mode_gauge:
        errors.append("USE_ASSB_AGING_INJECTION_ROHM requires LOCK_COMMON_MODE_GAUGE=True")
    if errors:
        raise RuntimeError("Invalid AgingFix1Config:\n- " + "\n- ".join(errors))
    return cfg


def save_aging_config(cfg: AgingFix1Config, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)


def load_aging_config(path: PathLike) -> AgingFix1Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return validate_aging_config(AgingFix1Config.from_dict(data))


def config_from_model_dir(model_dir: PathLike, *, fallback_params: Optional[Dict[str, Any]] = None) -> AgingFix1Config:
    """Load aging config from a model directory.

    Searches ``aging_config.json`` first and then ``config.json``.  If neither is
    available, parses *fallback_params* or defaults.
    """
    model_dir = Path(model_dir)
    for name in ("aging_config.json", "config.json"):
        p = model_dir / name
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if "use_assb_aging_fix1" in data or "USE_ASSB_AGING_FIX1" in data or name == "aging_config.json":
                # config.json from init_pinn may use uppercase keys.
                return parse_aging_fix1_config(data)
    return parse_aging_fix1_config(fallback_params or {})


def assert_fixed_material_identity(params: Optional[Dict[str, Any]] = None) -> None:
    """Guard against the old charge/discharge electrode-swap mistake.

    The ASSB effective-SPM convention is fixed material identity:
    a = Li-In/In negative electrode, c = NMC811 positive electrode.  Current sign
    changes flux and overpotential direction only.  This helper deliberately
    checks for dangerous parameter names that imply OCP/Ds/i0 swapping.
    """
    params = dict(params or {})
    forbidden_true = [
        "SWAP_ELECTRODES_BY_CURRENT",
        "SWAP_OCP_BY_CURRENT_SIGN",
        "SWITCH_MATERIAL_IDENTITY",
        "AGING_SWAP_OCP_BY_CURRENT",
    ]
    bad = [name for name in forbidden_true if _to_bool(params.get(name), False)]
    if bad:
        raise RuntimeError(
            "Fixed material identity violated. Do not switch OCP/Ds/i0/material identity by current sign: "
            + ", ".join(bad)
        )


__all__ = [
    "AgingFix1Config",
    "parse_aging_fix1_config",
    "validate_aging_config",
    "save_aging_config",
    "load_aging_config",
    "config_from_model_dir",
    "assert_fixed_material_identity",
]
