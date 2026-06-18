# -*- coding: utf-8 -*-
"""Synthetic import/forward/backward smoke for D17-P2 code only."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import sys
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from gv1.d17_pinn.config import load_config
from gv1.d17_pinn.trainer import train_smoke


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        t = np.linspace(0, 600, 32).astype(np.float32)
        I = (2.0 * np.sign(np.sin(2 * np.pi * t / 1800.0))).astype(np.float32)
        I[np.abs(np.sin(2 * np.pi * t / 1800.0)) < 0.15] = 0.0
        V = (3.45 + 0.35 * np.sin(2 * np.pi * t / 1800.0) + 0.03 * I).astype(np.float32)
        T = np.full_like(t, 25.0, dtype=np.float32)
        replay = root / "solution_replay_profile.npz"
        np.savez_compressed(replay, t_global_s=t, I_profile=I, voltage_exp=V, temperature_C=T)
        manifest = {"protocol":"D17-P2_SYNTHETIC_MANIFEST","manifest_hash_sha256":"synthetic","records":[{"canonical_cell_uid":"synthetic_profile_001","split":"train","replay_npz":str(replay),"is_flagged_probe":False}]}
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        cfg = load_config(None)
        cfg["paths"]["split_manifest"] = str(manifest_path)
        cfg["paths"]["resolved_spec"] = str(root / "missing_spec.json")
        cfg["paths"]["output_root"] = str(root / "out")
        cfg["train"].update({"split":"train","profile_index":0,"max_time_points":32,"time_window_s":600,"epochs":1,"device":"cpu","n_r":7})
        cfg["model"].update({"hidden_dim":8,"latent_hidden_dim":8})
        summary = train_smoke(cfg, root / "out" / "smoke_1profile_p2")
        print(json.dumps({"status":summary["status"],"reasons":summary.get("reasons"),"best_loss":summary.get("best_loss"),"final_voltage_mae_V":summary.get("final_metrics",{}).get("voltage_mae_V")}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
