from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from gv1.d18_cycleaware.array_io import discover_array_cases, load_array_case, load_split_index
from gv1.d18_cycleaware.common import expand_template
from gv1.d18_cycleaware.metrics import radial_deviation, regression_metrics, shell_volume_weights, volume_mean
from gv1.d18_cycleaware.model_scaffold import D18ModelConfig, synthetic_architecture_check


class ConfigAndSplitTests(unittest.TestCase):
    def test_nested_template_expansion(self) -> None:
        context = {"root": "E:/cache", "split": "${root}/split/manifest.json"}
        self.assertEqual(expand_template("${split}", context), "E:/cache/split/manifest.json")

    def test_nested_split_manifest_blocks_frozen_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            path = Path(tmp_text) / "split.json"
            path.write_text(
                '{"splits":{"frozen_test":[{"canonical_cell_uid":"Batch-6_battery-2"}],"train":["Batch-1_battery-1"]}}',
                encoding="utf-8",
            )
            index = load_split_index(path)
            frozen = index["batch6battery2"]
            train = index["batch1battery1"]
            self.assertEqual(frozen["split"], "frozen_test")
            self.assertEqual(train["split"], "train")


class MetricsTests(unittest.TestCase):
    def test_volume_projection_is_zero_mean(self) -> None:
        rho = np.linspace(0.0, 1.0, 17)
        field = np.random.default_rng(1).normal(size=(50, 17))
        delta = radial_deviation(field, rho)
        self.assertLess(float(np.max(np.abs(volume_mean(delta, rho)))), 1e-12)
        self.assertAlmostEqual(float(shell_volume_weights(rho, 17).sum()), 1.0, places=12)

    def test_regression_metrics_exact(self) -> None:
        x = np.linspace(-1.0, 1.0, 101)
        m = regression_metrics(x, x)
        self.assertAlmostEqual(m.mae, 0.0)
        self.assertAlmostEqual(m.r2, 1.0)
        self.assertAlmostEqual(m.corr, 1.0)


class ArchitectureTests(unittest.TestCase):
    def test_synthetic_architecture_contract(self) -> None:
        result = synthetic_architecture_check(D18ModelConfig(), batch_size=2, cycle_count=4, time_count=65)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["shape_ok"])
        self.assertLess(result["zero_volume_mean_max_abs_a"], 1e-5)
        self.assertLess(result["zero_volume_mean_max_abs_c"], 1e-5)


class ArrayLoadingTests(unittest.TestCase):
    def test_paired_npz_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            path = Path(tmp_text) / "Batch-1_2C_battery-1_prediction.npz"
            n, r = 200, 17
            t = np.arange(n, dtype=float)
            true = np.tile(np.linspace(0.2, 0.8, r), (n, 1)) + 0.01 * np.sin(t[:, None] / 15.0)
            np.savez_compressed(
                path,
                t_global_s=t,
                cycle_id=np.repeat(np.arange(1, 5), 50),
                split=np.asarray("train"),
                canonical_cell_uid=np.asarray("Batch-1_2C_battery-1"),
                protocol=np.asarray("2C"),
                semantic_branch=np.asarray("RG"),
                r_a=np.linspace(0, 1, r),
                cs_a_pred=true + 0.01,
                cs_a_true_report_only=true,
            )
            case = load_array_case(path, {}, 1024**3)
            self.assertEqual(case.n_time, n)
            self.assertIn("cs_a", case.available_states)
            self.assertEqual(case.pred["cs_a"].shape, (n, r))
            self.assertEqual(case.split, "train")


    def test_manifest_truth_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            root = Path(tmp_text)
            pred_path = root / "Batch-3_R2.5_battery-2_prediction.npz"
            truth_path = root / "solution_softlabels.npz"
            split_path = root / "split.json"
            n, r = 180, 17
            t = np.arange(n, dtype=float)
            true = np.tile(np.linspace(0.15, 0.75, r), (n, 1))
            np.savez_compressed(
                truth_path,
                t_global_s=t,
                cycle_id=np.repeat(np.arange(1, 4), 60),
                r_a=np.linspace(0, 1, r),
                cs_a=true,
            )
            np.savez_compressed(
                pred_path,
                t_global_s=t,
                cycle_id=np.repeat(np.arange(1, 4), 60),
                canonical_cell_uid=np.asarray("Batch-3_R2.5_battery-2"),
                protocol=np.asarray("R2.5"),
                semantic_branch=np.asarray("RG"),
                cs_a_pred=true + 0.002,
            )
            split_path.write_text(
                '{"records":[{"canonical_cell_uid":"Batch-3_R2.5_battery-2",'
                '"split":"validation","protocol":"R2.5","softlabel_npz":"'
                + str(truth_path).replace('\\', '\\\\')
                + '"}]}',
                encoding="utf-8",
            )
            case = load_array_case(pred_path, load_split_index(split_path), 1024**3)
            self.assertEqual(case.split, "validation")
            self.assertEqual(Path(case.truth_path), truth_path.resolve())
            self.assertTrue(np.allclose(case.true["cs_a"], true))

    def test_discovery_blocks_frozen_test_and_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            root = Path(tmp_text)
            n, r = 900, 17
            t = np.arange(n, dtype=float)
            y = np.tile(np.linspace(0.2, 0.8, r), (n, 1))
            for name, split in (("Batch-1_2C_battery-7_frozen.npz", "frozen_test"), ("Batch-1_2C_battery-8_unknown.npz", None)):
                kwargs = {
                    "t_global_s": t,
                    "cycle_id": np.repeat(np.arange(1, 10), 100),
                    "canonical_cell_uid": np.asarray(Path(name).stem),
                    "cs_a_pred": y,
                    "cs_a_true_report_only": y,
                }
                if split is not None:
                    kwargs["split"] = np.asarray(split)
                np.savez_compressed(root / name, **kwargs)
            config = {
                "paths": {"d17_split_manifest": str(root / "missing_split.json")},
                "s1": {
                    "prediction_roots": [str(root)],
                    "prediction_globs": ["**/*.npz"],
                    "allowed_splits": ["train", "validation", "internal_heldout"],
                    "blocked_splits": ["frozen_test", "test", "flagged_probe", "unknown"],
                    "require_dense": True,
                    "dense_min_time_points": 768,
                    "min_time_points": 128,
                    "max_cases": 10,
                },
            }
            result = discover_array_cases(config, root)
            self.assertEqual(result.cases, [])
            statuses = {row["status"] for row in result.inventory_rows}
            self.assertEqual(statuses, {"SKIP_BLOCKED_SPLIT"})


if __name__ == "__main__":
    unittest.main()
