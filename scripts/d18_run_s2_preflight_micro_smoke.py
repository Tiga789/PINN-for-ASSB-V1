from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d18_s2.common import dump_json, resolve_config, sha256_file, utc_now_iso
from gv1.d18_s2.data import build_micro_casepack
from gv1.d18_s2.losses import S2LossConfig
from gv1.d18_s2.model import S2ModelConfig, architecture_contract, synthetic_forward_check
from gv1.d18_s2.preflight import run_preflight
from gv1.d18_s2.trainer import S2TrainerConfig, run_micro_smoke


def log(message: str) -> None:
    print(f"[D18-S2] {message}", flush=True)


def backup_existing(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = path.with_name(path.name + f"_backup_{stamp}")
    counter = 1
    while target.exists():
        target = path.with_name(path.name + f"_backup_{stamp}_{counter:02d}")
        counter += 1
    path.rename(target)
    return target


def run(config_path: Path, project_root: Path, output_override: str | None = None) -> tuple[int, dict]:
    cfg = resolve_config(config_path, project_root=project_root)
    if output_override:
        cfg["paths"]["output_root"] = output_override
    output_root = Path(cfg["paths"]["output_root"])
    backup = backup_existing(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    copied_config = output_root / "D18_S2_RESOLVED_CONFIG.json"
    dump_json(cfg, copied_config)

    overall = {
        "stage": "D18-S2-PREFLIGHT-PLUS-MICRO-SMOKE",
        "created_at_utc": utc_now_iso(),
        "package_version": cfg.get("package_version"),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "prior_output_backup": str(backup) if backup else None,
        "formal_s2_training_enabled": False,
        "frozen_test_used": False,
        "flagged_probe_used": False,
        "final_goal": cfg.get("final_goal"),
    }
    try:
        log("1/5 validating architecture contract and physical output transform")
        model_cfg = S2ModelConfig.from_mapping(cfg["micro_smoke"].get("model"))
        architecture = architecture_contract(model_cfg)
        synthetic = synthetic_forward_check(model_cfg)
        dump_json(architecture, output_root / "D18_S2_ARCHITECTURE_CONTRACT.json")
        dump_json(synthetic, output_root / "D18_S2_ARCHITECTURE_SYNTHETIC_CHECK.json")
        if synthetic.get("status") != "PASS":
            raise RuntimeError("Architecture synthetic forward check failed")

        log("2/5 running exact-UID, split, provenance, source-grid, and environment preflight")
        preflight_dir = output_root / "d18_s2_preflight"
        preflight_result = run_preflight(
            cfg, project_root=project_root, output_dir=preflight_dir, progress=log
        )
        preflight_summary = preflight_result["summary"]
        if preflight_summary.get("status") != "PASS_PREFLIGHT_FOR_MICRO_SMOKE":
            raise RuntimeError("D18-S2 preflight failed; micro-smoke was not launched")

        log("3/5 building bounded micro-smoke casepack from source generator grids")
        casepack_dir = output_root / "d18_s2_micro_casepack"
        pre_cfg = cfg["preflight"]
        casepack = build_micro_casepack(
            preflight_result["records"],
            preflight_result["replay_paths"],
            output_dir=casepack_dir,
            min_source_points_per_cycle=int(pre_cfg["min_source_points_per_cycle"]),
            cycles_per_position=int(pre_cfg["cycles_per_position"]),
            micro_points_per_cycle=int(pre_cfg["micro_points_per_cycle"]),
            target_radial_points=int(pre_cfg["target_radial_points"]),
            progress=log,
        )
        if casepack.get("status") != "PASS":
            raise RuntimeError("Micro-smoke casepack build failed")

        log("4/5 running tiny cycle-aware operator training smoke (not formal S2 training)")
        loss_cfg = S2LossConfig.from_mapping(cfg["micro_smoke"].get("loss"))
        trainer_cfg = S2TrainerConfig.from_mapping(cfg["micro_smoke"].get("trainer"))
        if not trainer_cfg.disable_amp or not trainer_cfg.disable_torch_compile:
            raise RuntimeError("AMP and torch.compile must remain disabled in this package")
        micro_dir = output_root / "d18_s2_micro_smoke"
        micro = run_micro_smoke(
            casepack_profiles_dir=casepack_dir / "profiles",
            casepack_summary_path=casepack_dir / "profiles" / "D18_S2_MICRO_CASEPACK_SUMMARY.json",
            output_dir=micro_dir,
            model_config=model_cfg,
            loss_config=loss_cfg,
            trainer_config=trainer_cfg,
            progress=log,
        )

        log("5/5 writing bounded decision summary")
        completed = micro.get("status") in {"PASS_MICRO_SMOKE", "REVIEW_MICRO_SMOKE"}
        overall.update(
            {
                "status": (
                    "PASS_COMPLETED_MICRO_SMOKE"
                    if micro.get("status") == "PASS_MICRO_SMOKE"
                    else "REVIEW_COMPLETED_MICRO_SMOKE"
                ),
                "completed": completed,
                "architecture_status": synthetic.get("status"),
                "preflight_status": preflight_summary.get("status"),
                "casepack_status": casepack.get("status"),
                "micro_smoke_status": micro.get("status"),
                "micro_smoke_downsampled_view": True,
                "source_grid_counts_checked_without_downsampling": True,
                "teacher_initial_cbar_anchor_used": True,
                "formal_s2_training_eligible": False,
                "go_to_formal_s2_training": False,
                "go_to_s3": False,
                "config_sha256": sha256_file(copied_config),
                "next_action": (
                    "Upload the complete output directory for review. Do not start formal S2 training."
                ),
            }
        )
        dump_json(overall, output_root / "D18_S2_PREFLIGHT_MICRO_SMOKE_OVERALL_SUMMARY.json")
        log(f"completed with status={overall['status']}")
        return 0, overall
    except Exception as exc:
        overall.update(
            {
                "status": "FAIL_STOPPED_BEFORE_FORMAL_TRAINING",
                "completed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "formal_s2_training_eligible": False,
                "go_to_formal_s2_training": False,
                "next_action": "Inspect the failure report; do not bypass the failed preflight.",
            }
        )
        dump_json(overall, output_root / "D18_S2_PREFLIGHT_MICRO_SMOKE_OVERALL_SUMMARY.json")
        (output_root / "D18_S2_FAILURE_TRACEBACK.txt").write_text(overall["traceback"], encoding="utf-8")
        log(f"FAILED: {type(exc).__name__}: {exc}")
        return 2, overall


def main() -> int:
    parser = argparse.ArgumentParser(description="Run D18-S2 preflight and bounded micro-smoke")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "d18_s2_preflight_micro_smoke.json"),
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    code, _ = run(Path(args.config), Path(args.project_root).resolve(), args.output_root)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
