
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with p.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return default


def dump_json(data: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_text(text: str, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(text)


def sha256_file(path: str | Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def read_git(project_root: Path) -> Dict[str, Any]:
    def run(args: Sequence[str]) -> Dict[str, Any]:
        try:
            proc = subprocess.run(args, cwd=str(project_root), text=True, capture_output=True, timeout=20)
            return {"ok": proc.returncode == 0, "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
        except Exception as exc:
            return {"ok": False, "error": repr(exc)}
    return {
        "head": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "branch", "--show-current"]),
        "status_short": run(["git", "status", "--short"]),
    }


def collect_candidates(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for p in paths:
        if not p:
            continue
        p = Path(p)
        if not p.exists():
            continue
        if p.is_file():
            key = str(p.resolve())
            if key not in seen:
                out.append(p); seen.add(key)
        elif p.is_dir():
            # Only hash small/decision artifacts by default; avoid copying/scanning huge .npz/.pt trees.
            patterns = [
                "*.json", "*.csv", "*.md", "*.txt",
                "configs/*.json", "docs/*.txt", "model/*.pt", "model/*.pth"
            ]
            for pat in patterns:
                for q in sorted(p.glob(pat)):
                    if q.is_file():
                        key = str(q.resolve())
                        if key not in seen:
                            out.append(q); seen.add(key)
    return out


def hash_artifacts(paths: Iterable[Path], out_csv: Path, base_root: Optional[Path] = None, max_mb: float = 512.0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in collect_candidates(paths):
        try:
            size = p.stat().st_size
            too_large = size > max_mb * 1024 * 1024
            rows.append({
                "path": str(p),
                "relative_to_project": safe_rel(p, base_root) if base_root else "",
                "size_bytes": size,
                "sha256": "SKIPPED_TOO_LARGE" if too_large else sha256_file(p),
                "skipped_hash": bool(too_large),
            })
        except Exception as exc:
            rows.append({"path": str(p), "relative_to_project": "", "size_bytes": "", "sha256": "ERROR", "error": repr(exc), "skipped_hash": True})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "relative_to_project", "size_bytes", "sha256", "skipped_hash", "error"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def status_bool(d: Mapping[str, Any], key: str, expected: Any = True) -> bool:
    return d.get(key) == expected


def derive_release_decision(g0: Mapping[str, Any], g21: Mapping[str, Any], g3: Mapping[str, Any], g4: Mapping[str, Any]) -> Tuple[str, bool, List[str]]:
    reasons: List[str] = []
    if g0.get("status") != "PASS":
        reasons.append(f"G0 status is not PASS: {g0.get('status')}")
    if g0.get("g1_ready") is not True:
        reasons.append(f"G0 g1_ready is not true: {g0.get('g1_ready')}")
    if g21.get("status") != "PASS":
        reasons.append(f"G2.1 status is not PASS: {g21.get('status')}")
    if g21.get("g3_ready") is not True:
        reasons.append(f"G2.1 g3_ready is not true: {g21.get('g3_ready')}")
    if g3.get("status") != "PASS":
        reasons.append(f"G3 status is not PASS: {g3.get('status')}")
    if g3.get("promotion_status") != "PASS":
        reasons.append(f"G3 promotion_status is not PASS: {g3.get('promotion_status')}")
    if g3.get("g4_ready") is not True:
        reasons.append(f"G3 g4_ready is not true: {g3.get('g4_ready')}")
    if g4.get("status") != "PASS":
        reasons.append(f"G4 status is not PASS: {g4.get('status')}")
    if g4.get("final_candidate_ready") is not True:
        reasons.append(f"G4 final_candidate_ready is not true: {g4.get('final_candidate_ready')}")
    if g4.get("speed_status") not in ("PASS", None):
        reasons.append(f"G4 speed_status is not PASS: {g4.get('speed_status')}")
    return ("PASS" if not reasons else "REVIEW", not reasons, reasons)


def model_card_text(manifest: Mapping[str, Any]) -> str:
    metrics = manifest.get("key_metrics", {})
    policy = manifest.get("policy", {})
    return f"""# D17-G5 Final Candidate Model Card

## Candidate

- Candidate ID: `{manifest.get('candidate_id')}`
- Protocol: `{manifest.get('protocol')}`
- Status: `{manifest.get('status')}`
- Final release ready: `{manifest.get('final_release_ready')}`
- Recommendation: `{manifest.get('recommendation')}`

## Intended use

This candidate is a **D17-G P2Dlite-RG generator surrogate**. It is intended to quickly reproduce the D15 P2Dlite-RG model-consistent soft-label generator outputs for XJTU profiles under the frozen D17 train/validation/frozen-test split.

It should be described as a generator-distilled / supervised surrogate, not as direct experimental proof of true internal electrochemical states.

## Key metrics

- Frozen-test profile count: `{metrics.get('frozen_test_profile_count')}`
- Frozen-test mean R²: `{metrics.get('frozen_test_mean_r2')}`
- Frozen-test min R²: `{metrics.get('frozen_test_min_r2')}`
- Frozen-test phie min R²: `{metrics.get('frozen_test_phie_min_r2')}`
- Samples per second: `{metrics.get('samples_per_second')}`
- Speed status: `{metrics.get('speed_status')}`

## Protocol boundary

- Train-cell soft labels used for training: `{policy.get('train_cell_softlabels_used_for_training')}`
- Validation soft labels used only report-only: `{policy.get('validation_softlabels_report_only')}`
- Frozen-test soft labels used only in G3 report-only audit: `{policy.get('frozen_test_softlabels_report_only')}`
- Frozen-test feedback used to modify model: `{policy.get('frozen_test_feedback_used_to_modify_model')}`
- G5 training performed: `{policy.get('training_performed')}`

## Evidence chain

1. G0 audited generator semantics.
2. G1–G1.5R repaired supervised generator-surrogate training.
3. G2.1 repaired P4D/random-walk branch coverage and reached `g3_ready=true`.
4. G3 completed frozen-test report-only state audit with promotion PASS.
5. G4 completed final scorecard + speed audit with final candidate ready.
6. G5 freezes release and reproducibility artifacts.

## Important limitation

The D15 P2Dlite-RG labels are model-consistent soft labels, not experimentally measured internal states. This G5 candidate demonstrates held-out generator-surrogate performance against those labels, not independent physical truth.
"""


def reproducibility_text(args: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    return f"""# D17-G5 Reproducibility Notes

## Fixed candidate

Candidate ID: `{manifest.get('candidate_id')}`
Release created at UTC: `{manifest.get('created_at_utc')}`

## Required upstream artifacts

- G0 audit: `{args.get('g0_audit')}`
- G0 profile semantics CSV: `{args.get('g0_profile_semantics_csv')}`
- G2.1 summary: `{args.get('g21_summary')}`
- G2.1 directory: `{args.get('g21_dir')}`
- G3 summary: `{args.get('g3_summary')}`
- G3 scorecard: `{args.get('g3_scorecard')}`
- G3 directory: `{args.get('g3_dir')}`
- G4 final scorecard: `{args.get('g4_scorecard')}`
- G4 directory: `{args.get('g4_dir')}`
- Split manifest: `{args.get('split_manifest')}`

## No further tuning rule

After G3/G4 pass, G5 must not retrain, choose a new checkpoint, edit the split, or use frozen-test metrics to change the model. Any future improvements must start a new explicitly named branch and must not overwrite this release.

## Suggested citation phrase in reports

`D17-G4/G5 is a P2Dlite-RG generator-surrogate candidate. It achieved held-out frozen-test report-only soft-label R² metrics against D15 P2Dlite-RG model-consistent labels. These labels are not direct experimental internal-state truth.`
"""


def final_report_text(manifest: Mapping[str, Any]) -> str:
    reasons = manifest.get("reasons", [])
    metrics = manifest.get("key_metrics", {})
    return f"""# D17-G5 Final Release Report

## Decision

`{manifest.get('status')}` — `{manifest.get('recommendation')}`

Final release ready: `{manifest.get('final_release_ready')}`

Reasons / blockers:

{os.linesep.join('- ' + str(r) for r in reasons) if reasons else '- None'}

## Final metrics copied from G3/G4

| Metric | Value |
|---|---:|
| frozen_test_profile_count | {metrics.get('frozen_test_profile_count')} |
| frozen_test_mean_r2 | {metrics.get('frozen_test_mean_r2')} |
| frozen_test_min_r2 | {metrics.get('frozen_test_min_r2')} |
| frozen_test_phie_min_r2 | {metrics.get('frozen_test_phie_min_r2')} |
| samples_per_second | {metrics.get('samples_per_second')} |
| speed_status | {metrics.get('speed_status')} |

## Frozen candidate statement

The current D17-G generator surrogate should be frozen as `{manifest.get('candidate_id')}` if and only if `final_release_ready=true`. The candidate is not a strict no-state-label inverse PINN; it is a train-cell soft-label supervised, generator-distilled surrogate with validation report-only and frozen-test report-only evaluation.

## Artifacts

See:

- `D17_G5_FINAL_RELEASE_MANIFEST.json`
- `D17_G5_ARTIFACT_HASHES.csv`
- `D17_G5_MODEL_CARD.md`
- `D17_G5_REPRODUCIBILITY_NOTES.md`
"""


def run_g5(
    *,
    project_root: str | Path,
    out_dir: str | Path,
    split_manifest: str | Path,
    g0_audit: str | Path,
    g0_profile_semantics_csv: str | Path,
    g21_summary: str | Path,
    g21_dir: str | Path,
    g3_summary: str | Path,
    g3_scorecard: str | Path,
    g3_dir: str | Path,
    g4_scorecard: str | Path,
    g4_dir: str | Path,
    no_state_label_audit: str | Path | None = None,
    copy_small_artifacts: bool = True,
    max_hash_mb: float = 512.0,
) -> Dict[str, Any]:
    project_root = Path(project_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g0 = load_json(g0_audit, {})
    g21 = load_json(g21_summary, {})
    g3 = load_json(g3_summary, {})
    g3_sc = load_json(g3_scorecard, {})
    g4 = load_json(g4_scorecard, {})
    split = load_json(split_manifest, {})
    no_state = load_json(no_state_label_audit, {}) if no_state_label_audit else {}

    status, ready, reasons = derive_release_decision(g0, g21, g3, g4)

    key_metrics = {
        "frozen_test_profile_count": g3.get("frozen_test_profile_count") or g4.get("frozen_test_profile_count"),
        "frozen_test_mean_r2": g3.get("frozen_test_mean_r2") or g4.get("frozen_test_mean_r2"),
        "frozen_test_min_r2": g3.get("frozen_test_min_r2") or g4.get("frozen_test_min_r2"),
        "frozen_test_phie_min_r2": g3.get("frozen_test_phie_min_r2") or g4.get("frozen_test_phie_min_r2"),
        "samples_per_second": g4.get("samples_per_second"),
        "speed_status": g4.get("speed_status"),
    }
    candidate_id = "D17-G4_GENERATOR_SURROGATE_CANDIDATE__G5_RELEASE"

    artifact_hash_csv = out_dir / "D17_G5_ARTIFACT_HASHES.csv"
    artifact_paths: List[Path] = [
        Path(split_manifest), Path(g0_audit), Path(g0_profile_semantics_csv),
        Path(g21_summary), Path(g21_dir), Path(g3_summary), Path(g3_scorecard), Path(g3_dir),
        Path(g4_scorecard), Path(g4_dir),
    ]
    if no_state_label_audit:
        artifact_paths.append(Path(no_state_label_audit))
    hashes = hash_artifacts(artifact_paths, artifact_hash_csv, base_root=project_root, max_mb=max_hash_mb)

    manifest: Dict[str, Any] = {
        "protocol": "D17-G5_FINAL_RELEASE_REPRODUCIBILITY_FREEZE",
        "created_at_utc": utc_now(),
        "status": status,
        "final_release_ready": ready,
        "candidate_id": candidate_id,
        "recommendation": "FREEZE_AND_ARCHIVE_D17_G_GENERATOR_SURROGATE" if ready else "REVIEW_G5_BLOCKERS_BEFORE_RELEASE",
        "reasons": reasons,
        "policy": {
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "model_modified": False,
            "split_modified": False,
            "frozen_test_feedback_used_to_modify_model": False,
            "train_cell_softlabels_used_for_training": True,
            "validation_softlabels_report_only": True,
            "frozen_test_softlabels_report_only": True,
            "not_experimental_internal_state_truth": True,
        },
        "git": read_git(project_root),
        "split_manifest": {
            "path": str(split_manifest),
            "manifest_hash_sha256": split.get("manifest_hash_sha256"),
            "counts": split.get("counts"),
        },
        "prerequisites": {
            "g0": {"path": str(g0_audit), "status": g0.get("status"), "g1_ready": g0.get("g1_ready")},
            "g21": {"path": str(g21_summary), "status": g21.get("status"), "g3_ready": g21.get("g3_ready")},
            "g3": {"path": str(g3_summary), "status": g3.get("status"), "promotion_status": g3.get("promotion_status"), "g4_ready": g3.get("g4_ready")},
            "g4": {"path": str(g4_scorecard), "status": g4.get("status"), "final_candidate_ready": g4.get("final_candidate_ready"), "speed_status": g4.get("speed_status")},
            "no_state_label_audit": {"path": str(no_state_label_audit) if no_state_label_audit else "", "pass": no_state.get("pass") or no_state.get("status")},
        },
        "key_metrics": key_metrics,
        "artifact_hashes_csv": str(artifact_hash_csv),
        "artifact_hash_count": len(hashes),
        "output_files": {},
    }

    manifest_path = out_dir / "D17_G5_FINAL_RELEASE_MANIFEST.json"
    model_card_path = out_dir / "D17_G5_MODEL_CARD.md"
    repro_path = out_dir / "D17_G5_REPRODUCIBILITY_NOTES.md"
    report_path = out_dir / "D17_G5_FINAL_RELEASE_REPORT.md"
    dump_json(manifest, manifest_path)
    write_text(model_card_text(manifest), model_card_path)
    write_text(reproducibility_text({
        "g0_audit": str(g0_audit), "g0_profile_semantics_csv": str(g0_profile_semantics_csv),
        "g21_summary": str(g21_summary), "g21_dir": str(g21_dir),
        "g3_summary": str(g3_summary), "g3_scorecard": str(g3_scorecard), "g3_dir": str(g3_dir),
        "g4_scorecard": str(g4_scorecard), "g4_dir": str(g4_dir),
        "split_manifest": str(split_manifest),
    }, manifest), repro_path)
    write_text(final_report_text(manifest), report_path)

    manifest["output_files"] = {
        "manifest_json": str(manifest_path),
        "model_card_md": str(model_card_path),
        "reproducibility_notes_md": str(repro_path),
        "final_report_md": str(report_path),
        "artifact_hashes_csv": str(artifact_hash_csv),
    }
    dump_json(manifest, manifest_path)

    if copy_small_artifacts:
        copied_csv = out_dir / "D17_G5_COPIED_SMALL_ARTIFACTS.csv"
        copied_rows = []
        copy_root = out_dir / "frozen_small_artifacts"
        for row in hashes:
            p = Path(row.get("path", ""))
            if not p.is_file():
                continue
            try:
                size = p.stat().st_size
                if size > 20 * 1024 * 1024:
                    continue
                dest = copy_root / p.name
                # Avoid overwrite collisions by prefixing parent if needed.
                if dest.exists():
                    dest = copy_root / f"{p.parent.name}__{p.name}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(p, dest)
                copied_rows.append({"source": str(p), "copied_to": str(dest), "size_bytes": size, "sha256": row.get("sha256")})
            except Exception as exc:
                copied_rows.append({"source": str(p), "copied_to": "ERROR", "size_bytes": "", "sha256": "", "error": repr(exc)})
        with copied_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "copied_to", "size_bytes", "sha256", "error"], extrasaction="ignore")
            writer.writeheader(); writer.writerows(copied_rows)
        manifest["output_files"]["copied_small_artifacts_csv"] = str(copied_csv)
        dump_json(manifest, manifest_path)

    return manifest
