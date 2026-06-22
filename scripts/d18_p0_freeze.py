from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from gv1.d18_cycleaware.common import (  # noqa: E402
    collect_git_state,
    compact_exception,
    dump_json,
    expand_candidate_paths,
    iter_files,
    load_json,
    resolve_config_path,
    resolve_project_root,
    safe_relpath,
    sha256_file,
    stable_json_sha256,
    tree_fingerprint,
    utc_now_iso,
    write_csv,
)


def _fingerprint_path(path: Path, item: Mapping[str, Any]) -> dict[str, Any]:
    if path.is_file():
        stat = path.stat()
        return {
            "exists": True,
            "path": str(path),
            "kind": "file",
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": sha256_file(path),
        }
    return tree_fingerprint(
        path,
        include_globs=[str(x) for x in item.get("include_globs", ["**/*"])],
        exclude_globs=[str(x) for x in item.get("exclude_globs", ["**/__pycache__/**", "**/*.pyc"])],
        full_hash_max_bytes=int(item.get("full_hash_max_bytes", 512 * 1024 * 1024)),
        max_files=int(item.get("max_files", 20_000)),
    )


def _copy_small_evidence(
    source: Path,
    destination_root: Path,
    item: Mapping[str, Any],
    max_bytes: int,
) -> list[str]:
    copied: list[str] = []
    if not bool(item.get("snapshot_small_files", True)):
        return copied
    if source.is_file():
        files = [source]
        base = source.parent
    else:
        files = list(
            iter_files(
                source,
                [str(x) for x in item.get("include_globs", ["**/*.json", "**/*.csv", "**/*.md", "**/*.txt"])],
                [str(x) for x in item.get("exclude_globs", ["**/__pycache__/**", "**/*.pyc"])],
            )
        )
        base = source
    for file_path in files:
        try:
            if file_path.stat().st_size > max_bytes:
                continue
            relative = Path(safe_relpath(file_path, base))
            target = destination_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target)
            copied.append(target.as_posix())
        except OSError:
            continue
    return copied


def run_p0(config_path: str | Path, output_root_override: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_json(config_path)
    project_root = resolve_project_root(config_path, config)
    paths = config.get("paths", {}) if isinstance(config.get("paths"), Mapping) else {}
    raw_output = output_root_override or paths.get("output_root", "D18_Output")
    output_root = (
        Path(output_root_override).resolve()
        if output_root_override is not None
        else resolve_config_path(str(raw_output), config, project_root)
    )
    out = output_root / "d18_p0_freeze"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[D18-P0] project_root={project_root}", flush=True)
    print(f"[D18-P0] output_dir={out}", flush=True)

    p0 = config.get("p0", {}) if isinstance(config.get("p0"), Mapping) else {}
    artifacts = p0.get("artifacts", []) if isinstance(p0.get("artifacts", []), list) else []
    snapshot_max = int(p0.get("copy_evidence_max_bytes", 8 * 1024 * 1024))
    git_state = collect_git_state(project_root)
    dump_json(git_state, out / "p0_git_state.json")

    rows: list[dict[str, Any]] = []
    manifest_items: list[dict[str, Any]] = []
    for index, item_raw in enumerate(artifacts, start=1):
        item = item_raw if isinstance(item_raw, Mapping) else {}
        artifact_id = str(item.get("id", f"artifact_{index:02d}"))
        required = bool(item.get("required", False))
        candidates = [str(x) for x in item.get("candidates", [])]
        print(f"[D18-P0] {index}/{len(artifacts)} freeze {artifact_id}", flush=True)
        matches = [p for p in expand_candidate_paths(candidates, config, project_root) if p.exists()]
        artifact_entry: dict[str, Any] = {
            "id": artifact_id,
            "description": str(item.get("description", "")),
            "required": required,
            "candidate_templates": candidates,
            "matched_paths": [str(p) for p in matches],
            "fingerprints": [],
            "snapshot_files": [],
        }
        status = "FOUND" if matches else ("MISSING_REQUIRED" if required else "MISSING_OPTIONAL")
        try:
            for match_index, match in enumerate(matches):
                artifact_entry["fingerprints"].append(_fingerprint_path(match, item))
                copied = _copy_small_evidence(
                    match,
                    out / "evidence_snapshot" / artifact_id / f"match_{match_index:02d}",
                    item,
                    snapshot_max,
                )
                artifact_entry["snapshot_files"].extend(copied)
        except Exception as exc:
            status = "ERROR"
            artifact_entry["error"] = compact_exception(exc)
        artifact_entry["status"] = status
        manifest_items.append(artifact_entry)
        rows.append(
            {
                "id": artifact_id,
                "required": required,
                "status": status,
                "match_count": len(matches),
                "matched_paths": [str(p) for p in matches],
                "description": artifact_entry["description"],
            }
        )

    required_missing = [r for r in rows if r["required"] and r["status"] != "FOUND"]
    artifact_errors = [r for r in rows if r["status"] == "ERROR"]
    project_cfg = config.get("project", {}) if isinstance(config.get("project"), Mapping) else {}
    expected_short = str(project_cfg.get("expected_remote_short_sha", "")).strip().lower()
    local_short = str(
        git_state.get("commands", {}).get("short_head", {}).get("stdout", "")
    ).strip().lower()
    git_match = bool(expected_short and local_short.startswith(expected_short))
    if not expected_short:
        git_status = "RECORDED_NO_PINNED_SHA"
    elif git_match:
        git_status = "MATCH_EXPECTED_REMOTE"
    else:
        git_status = "REVIEW_LOCAL_GIT_STATE"

    status = "PASS" if not required_missing and not artifact_errors else "REVIEW"
    manifest = {
        "stage": "D18-P0",
        "created_at_utc": utc_now_iso(),
        "status": status,
        "training_launched": False,
        "project_root": str(project_root),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "manifest_schema": "d18_p0_freeze_manifest_v1",
        "expected_remote_short_sha": expected_short,
        "local_short_head": local_short,
        "git_alignment_status": git_status,
        "required_missing_count": len(required_missing),
        "artifact_error_count": len(artifact_errors),
        "artifact_count": len(manifest_items),
        "artifacts": manifest_items,
    }
    manifest["manifest_content_sha256"] = stable_json_sha256(manifest)
    dump_json(manifest, out / "p0_freeze_manifest.json")
    write_csv(rows, out / "p0_artifact_status.csv")
    lines = [
        "# D18-P0 Freeze Status",
        "",
        f"- Status: **{status}**",
        f"- Training launched: **False**",
        f"- Required artifacts missing: **{len(required_missing)}**",
        f"- Git alignment: **{git_status}**",
        f"- Local short HEAD: `{local_short or 'unavailable'}`",
        f"- Expected GitHub short SHA at package build: `{expected_short or 'not set'}`",
        "",
        "P0 records hashes and small evidence files only. It does not copy or regenerate the 52+ GB ALL55 teacher dataset.",
    ]
    (out / "P0_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[D18-P0] status={status} required_missing={len(required_missing)}", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="D18-P0 freeze D17 evidence without training")
    parser.add_argument("--config", default="configs/d18_p0_s0_s1.json")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    result = run_p0(args.config, args.output_root)
    return 0 if result["status"] in {"PASS", "REVIEW"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
