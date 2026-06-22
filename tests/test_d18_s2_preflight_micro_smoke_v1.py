from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d18_s2.common import dump_json
from gv1.d18_s2.data import safe_time_gradient, select_position_cycles, stratified_cycle_indices
from gv1.d18_s2.losses import S2LossConfig, compute_loss
from gv1.d18_s2.model import CycleAwareS2Operator, S2ModelConfig, ZeroMeanRadialBasis, synthetic_forward_check
from gv1.d18_s2.uid import load_role_index, parse_canonical_uid, path_mentions_exact_uid


class TestD18S2PreflightMicroSmoke(unittest.TestCase):
    def test_01_canonical_uid_exact_numeric_tokens(self) -> None:
        one = parse_canonical_uid("Batch-2_3C_battery-1")
        ten = parse_canonical_uid("Batch-2_3C_battery-10")
        self.assertNotEqual(one, ten)
        self.assertEqual(one.battery, 1)
        self.assertEqual(ten.battery, 10)

    def test_02_path_guard_does_not_confuse_battery_1_and_10(self) -> None:
        uid = parse_canonical_uid("Batch-2_3C_battery-1")
        self.assertFalse(path_mentions_exact_uid("X:/replay/Batch-2_3C_battery-10/solution_replay_profile.npz", uid))
        self.assertTrue(path_mentions_exact_uid("X:/replay/Batch-2_3C_battery-1/solution_replay_profile.npz", uid))

    def test_03_role_index_accepts_locked_g2_split_column(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "roles.csv"
            with p.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["split", "canonical_cell_uid"])
                writer.writeheader()
                writer.writerow({"split": "G2_train_fit", "canonical_cell_uid": "Batch-1_2C_battery-1"})
            index = load_role_index(p)
        self.assertEqual(index["batch-1_2c_battery-1"]["role"], "fit_train")

    def test_04_zero_mean_basis_is_pointwise_bounded(self) -> None:
        basis = ZeroMeanRadialBasis(17, 6)
        coef = torch.randn(5, 11, 6)
        shape = basis(coef)
        self.assertLessEqual(float(torch.max(torch.abs(shape))), 1.000001)
        self.assertLess(float(torch.max(torch.abs(basis.weighted_mean(shape)))), 1e-5)

    def test_05_synthetic_architecture_forward_passes(self) -> None:
        result = synthetic_forward_check(S2ModelConfig(cycle_hidden_dim=16, local_hidden_dim=16, fused_hidden_dim=24))
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(result["checks"].values()))

    def test_06_safe_gradient_handles_repeated_timestamps(self) -> None:
        t = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        grad = safe_time_gradient(y, t)
        self.assertTrue(np.isfinite(grad).all())
        self.assertEqual(grad.shape, y.shape)

    def test_07_cycle_position_selection_has_no_overlap(self) -> None:
        selected, positions = select_position_cycles(np.arange(1, 31), 2)
        self.assertEqual(selected.size, 6)
        self.assertEqual(np.unique(selected).size, 6)
        self.assertEqual(set(positions.values()), {"early", "middle", "late"})

    def test_08_stratified_cycle_sampling_returns_exact_count(self) -> None:
        idx = np.arange(120)
        phase = np.array(["charge"] * 40 + ["rest"] * 20 + ["discharge"] * 45 + ["rest"] * 15)
        selected = stratified_cycle_indices(idx, phase, 32)
        self.assertEqual(selected.size, 32)
        self.assertEqual(np.unique(selected).size, 32)
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], 119)

    def test_09_json_writer_never_emits_nan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.json"
            dump_json({"x": float("nan"), "y": float("inf")}, p)
            text = p.read_text(encoding="utf-8")
            parsed = json.loads(text)
        self.assertIsNone(parsed["x"])
        self.assertIsNone(parsed["y"])
        self.assertNotIn("NaN", text)

    def test_10_loss_is_finite_for_valid_model_output(self) -> None:
        cfg = S2ModelConfig(cycle_hidden_dim=8, local_hidden_dim=8, fused_hidden_dim=12, branch_embed_dim=2, radial_basis_count=2)
        model = CycleAwareS2Operator(cfg)
        b, c, ppc = 2, 3, 8
        t = c * ppc
        cycle_index = torch.arange(c).repeat_interleave(ppc)[None].repeat(b, 1)
        batch = {
            "cycle_features": torch.randn(b, c, 20),
            "local_features": torch.randn(b, t, 14),
            "cycle_index": cycle_index,
            "cbar_baseline": torch.rand(b, t, 2) * 1000 + torch.tensor([16000.0, 32000.0]),
            "potential_baseline": torch.randn(b, t, 2) * 0.01 + torch.tensor([0.0, 3.7]),
            "branch_id": torch.tensor([0, 1]),
            "theta_offset": torch.zeros(b, 2),
            "theta_scale": torch.tensor([[1 / 32000.0, 1 / 51000.0]]).repeat(b, 1),
            "selected_cycle_ids": torch.tensor([[1, 2, 3], [1, 2, 3]]),
        }
        out = model(**{k: batch[k] for k in ("cycle_features", "local_features", "cycle_index", "cbar_baseline", "potential_baseline", "branch_id", "theta_offset", "theta_scale")})
        for state in ("cs_a", "cs_c", "theta_a", "theta_c", "phie", "phis_c"):
            batch[f"{state}_true"] = out[state].detach() + 0.001
        batch["cbar_true_report_only"] = torch.cat([out["cbar_a"], out["cbar_c"]], dim=-1).detach()
        scales = {"cs_a": 1000.0, "cs_c": 1000.0, "theta_a": 0.1, "theta_c": 0.1, "phie": 0.1, "phis_c": 1.0}
        loss, parts = compute_loss(out, batch, scales, S2LossConfig())
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIn("boundary_jump_match", parts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
