from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Keep the installer selftest deterministic and prevent BLAS/OpenMP thread
# oversubscription on workstations that already have a training environment active.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d18_s2.common import dump_json
from gv1.d18_s2.uid import parse_canonical_uid
from scripts.d18_run_s2_preflight_micro_smoke import run

FIT = [
    "Batch-1_2C_battery-1",
    "Batch-2_3C_battery-5",
    "Batch-3_R2.5_battery-2",
    "Batch-4_R3_battery-4",
    "Batch-5_random_walk_battery-2",
    "Batch-6_GEO_battery-1",
]
INTERNAL = ["Batch-1_2C_battery-3", "Batch-6_GEO_battery-5"]
VALIDATION = ["Batch-2_3C_battery-10", "Batch-5_random_walk_battery-3"]
ALL = FIT + INTERNAL + VALIDATION
EXTRA_COLLISION_RECORDS = ["Batch-2_3C_battery-1"]


def git_head(root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Selftest project root must be a git checkout: {root}: {proc.stderr}")
    return proc.stdout.strip()


def create_profile(soft_path: Path, replay_path: Path, uid_text: str, profile_index: int) -> None:
    uid = parse_canonical_uid(uid_text)
    cycles = np.arange(1, 7, dtype=np.int64)
    points = 96
    cycle_id = np.repeat(cycles, points)
    n = cycle_id.size
    time = np.arange(n, dtype=np.float64)
    local = np.tile(np.linspace(0.0, 1.0, points), cycles.size)
    phase = np.empty(n, dtype="U16")
    current = np.zeros(n, dtype=np.float64)
    for k in range(cycles.size):
        sl = slice(k * points, (k + 1) * points)
        x = local[sl]
        amp = 2.0 + 0.15 * (profile_index % 4)
        current[sl] = np.where(x < 0.38, amp, np.where(x < 0.50, 0.0, np.where(x < 0.90, -0.75 * amp, 0.0)))
        phase[sl] = np.where(x < 0.38, "charge", np.where(x < 0.50, "rest", np.where(x < 0.90, "discharge", "rest")))
    q = np.cumsum(current) / 3600.0
    q_abs = np.cumsum(np.abs(current)) / 3600.0
    cyc_norm = (cycle_id - 1) / 5.0
    voltage = 3.55 + 0.42 * np.sin(2 * np.pi * local - 0.5) + 0.035 * current - 0.04 * cyc_norm
    temperature = 25.0 + 0.4 * np.abs(current) + 0.15 * np.sin(2 * np.pi * local)
    r = np.linspace(0.0, 1.0, 17, dtype=np.float64)
    radial = (r**2 - 0.6).reshape(1, -1)
    cbar_a = 17000.0 + 1500.0 * q + 25.0 * profile_index
    cbar_c = 33000.0 - 2200.0 * q - 30.0 * profile_index
    grad_a = (180.0 * np.tanh(current / 2.0))[:, None] * radial
    grad_c = (-240.0 * np.tanh(current / 2.0))[:, None] * radial
    cs_a = cbar_a[:, None] + grad_a
    cs_c = cbar_c[:, None] + grad_c
    theta_a = cs_a / 32000.0
    theta_c = cs_c / 51000.0
    phie = (-0.015 + 0.018 * np.tanh(current / 2.0) + 0.004 * np.sin(2 * np.pi * local))[:, None]
    phis_c = (voltage + 0.015 * np.sin(4 * np.pi * local) + 0.002 * profile_index)[:, None]
    soft_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        soft_path,
        t_global_s=time,
        cycle_id=cycle_id,
        I_profile=current,
        voltage_exp=voltage,
        temperature_C=temperature,
        step_type=phase,
        r_a=r,
        r_c=r,
        cs_a=cs_a.astype(np.float32),
        cs_c=cs_c.astype(np.float32),
        theta_a=theta_a.astype(np.float32),
        theta_c=theta_c.astype(np.float32),
        phie=phie.astype(np.float32),
        phis_c=phis_c.astype(np.float32),
        canonical_cell_uid=np.array(uid.canonical),
        cell_uid=np.array(uid.cell_uid),
        batch=np.array(f"Batch-{uid.batch}"),
        protocol=np.array(uid.protocol),
    )
    np.savez_compressed(
        replay_path,
        t_global_s=time,
        cycle_id=cycle_id,
        I_profile=current,
        voltage_exp=voltage,
        temperature_C=temperature,
        step_type=phase,
        canonical_cell_uid=np.array(uid.canonical),
        cell_uid=np.array(uid.cell_uid),
        batch=np.array(f"Batch-{uid.batch}"),
        protocol=np.array(uid.protocol),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    project_root = Path(args.project_root).resolve()
    expected = git_head(project_root)[:7]
    temp_ctx = tempfile.TemporaryDirectory(prefix="d18_s2_selftest_")
    work = Path(temp_ctx.name)
    data_root = work / "data"
    soft_root = data_root / "softlabels" / "profiles"
    replay_root = data_root / "replays"
    split_path = data_root / "d17_split_manifest.json"
    role_path = data_root / "D17_G2_PREDICTION_MANIFEST.csv"
    prior = data_root / "prior"
    prior.mkdir(parents=True)
    dump_json({"status": "PASS_READY_FOR_HUMAN_ARCHITECTURE_REVIEW"}, prior / "overall.json")
    dump_json({"status": "PASS"}, prior / "s0.json")
    dump_json({"status": "PASS_VALID_DIAGNOSTIC_COVERAGE", "frozen_test_used": False}, prior / "s1.json")
    dump_json({"status": "PASS"}, prior / "coverage.json")

    records = []
    role_rows = []
    for idx, uid_text in enumerate(ALL + EXTRA_COLLISION_RECORDS):
        uid = parse_canonical_uid(uid_text)
        soft = soft_root / uid.cell_uid / "solution_softlabels.npz"
        replay = replay_root / uid.canonical / "solution_replay_profile.npz"
        create_profile(soft, replay, uid_text, idx)
        declared_replay = replay
        # Deliberately inject the historical battery-1 -> battery-10 provenance bug.
        if uid_text == "Batch-2_3C_battery-1":
            declared_replay = replay_root / "Batch-2_3C_battery-10" / "solution_replay_profile.npz"
        split = "validation" if uid_text in VALIDATION else "train"
        records.append(
            {
                "cell_uid": uid.cell_uid,
                "canonical_cell_uid": uid.canonical,
                "batch": f"Batch-{uid.batch}",
                "protocol": uid.protocol,
                "battery": f"battery-{uid.battery}",
                "softlabel_dir": str(soft.parent),
                "softlabel_npz": str(soft),
                "softlabel_summary": "",
                "source_stage": "synthetic_selftest",
                "resolved_spec_hash": "",
                "duplicate_candidates": 0,
                "replay_npz": str(declared_replay),
                "is_flagged_probe": False,
                "split": split,
            }
        )
        if uid_text in ALL:
            role_split = "G2_train_fit" if uid_text in FIT else ("G2_train_internal_heldout" if uid_text in INTERNAL else "G2_validation_report_only")
            role_rows.append({"split": role_split, "index": idx, "canonical_cell_uid": uid.canonical, "pred_npz": "", "n_time": 512})
    dump_json({"protocol": "D17-P1_SPLIT_MANIFEST", "replay_roots": [str(replay_root)], "records": records}, split_path)
    with role_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "index", "canonical_cell_uid", "pred_npz", "n_time"])
        writer.writeheader(); writer.writerows(role_rows)

    config = {
        "package_version": "D18-S2-SYNTHETIC-SELFTEST",
        "expected_github_commit": expected,
        "human_review_token": "D18_S1_REVIEW_ACCEPTED_FOR_S2_PREFLIGHT_ONLY_20260618",
        "formal_s2_training_enabled": False,
        "final_goal": "synthetic_test_only",
        "paths": {
            "output_root": str(work / "output"),
            "prior_s0_s1_overall_summary": str(prior / "overall.json"),
            "prior_s0_validation": str(prior / "s0.json"),
            "prior_s1_summary": str(prior / "s1.json"),
            "prior_s1_coverage": str(prior / "coverage.json"),
            "d17_split_manifest": str(split_path),
            "d17_g2_prediction_manifest": str(role_path),
        },
        "preflight": {
            "fit_train_uids": FIT,
            "internal_heldout_uids": INTERNAL,
            "validation_report_only_uids": VALIDATION,
            "additional_replay_roots": [],
            "min_source_points_per_cycle": 96,
            "cycles_per_position": 2,
            "micro_points_per_cycle": 4,
            "target_radial_points": 17,
            "minimum_free_disk_bytes": 1000000,
        },
        "micro_smoke": {
            "model": {"cycle_hidden_dim": 4, "local_hidden_dim": 4, "fused_hidden_dim": 8, "branch_embed_dim": 2, "radial_basis_count": 2},
            "loss": {},
            "trainer": {"device": "cpu", "epochs": 1, "batch_size_profiles": 6, "learning_rate": 0.003, "min_relative_train_improvement": -1.0, "disable_amp": True, "disable_torch_compile": True, "save_predictions": False},
        },
    }
    config_path = work / "selftest_config.json"
    dump_json(config, config_path)
    code, overall = run(config_path, project_root)
    summary_path = work / "output" / "D18_S2_PREFLIGHT_MICRO_SMOKE_OVERALL_SUMMARY.json"
    if code != 0 or not summary_path.exists():
        print(json.dumps(overall, indent=2), file=sys.stderr)
        preflight_debug = work / "output" / "d18_s2_preflight" / "D18_S2_PREFLIGHT_SUMMARY.json"
        if preflight_debug.exists():
            print(preflight_debug.read_text(encoding="utf-8"), file=sys.stderr)
        if args.keep:
            print(f"kept at {work}", file=sys.stderr)
            temp_ctx._finalizer.detach()  # type: ignore[attr-defined]
        else:
            temp_ctx.cleanup()
        return 2
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    required = {
        "architecture_status": "PASS",
        "preflight_status": "PASS_PREFLIGHT_FOR_MICRO_SMOKE",
        "casepack_status": "PASS",
    }
    for key, expected_value in required.items():
        if result.get(key) != expected_value:
            raise RuntimeError(f"selftest {key}: {result.get(key)!r} != {expected_value!r}")
    collision = json.loads((work / "output" / "d18_s2_preflight" / "D18_S2_PREFLIGHT_SUMMARY.json").read_text(encoding="utf-8"))["battery1_battery10_collision_audit"]
    if not all(collision.values()):
        raise RuntimeError(f"battery-1/battery-10 collision guard failed: {collision}")
    print("D18-S2 synthetic end-to-end selftest: PASS")
    print(f"result={result['status']}")
    if args.keep:
        print(f"kept at {work}")
        temp_ctx._finalizer.detach()  # type: ignore[attr-defined]
    else:
        temp_ctx.cleanup()
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
