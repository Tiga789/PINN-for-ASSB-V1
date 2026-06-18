#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, csv, json, math, sys, time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []; seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fields.append(k); seen.add(k)
    if not fields:
        fields = ["empty"]; rows = [{"empty": ""}]
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow(dict(r))


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def compact_float(x: Any, digits: int = 6) -> str:
    try:
        v = float(x)
        return f"{v:.{digits}g}" if math.isfinite(v) else "nan"
    except Exception:
        return "nan"


def hhmmss(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"


def canonical_uid(r: Mapping[str, Any]) -> str:
    return str(r.get("canonical_cell_uid") or r.get("cell_uid") or r.get("cell_id") or "")


def protocol_of(r: Mapping[str, Any]) -> str:
    p = str(r.get("protocol") or "")
    if p: return p
    uid = canonical_uid(r)
    for x in ["random_walk", "GEO", "R2.5", "R3", "3C", "2C"]:
        if x in uid: return x
    return "UNKNOWN"


def load_semantics(path: str | Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in read_csv_rows(path):
        for k in ["canonical_cell_uid", "cell_uid", "profile_id"]:
            v = str(row.get(k, "")).strip()
            if v: out[v] = row
        sp = str(row.get("softlabel_npz", "")).strip()
        if sp:
            out[sp] = row
            try: out[Path(sp).parent.name] = row
            except Exception: pass
    return out


def branch_of(r: Mapping[str, Any], sem_map: Mapping[str, Mapping[str, str]]) -> str:
    for k in [canonical_uid(r), str(r.get("cell_uid") or ""), str(r.get("softlabel_npz") or "")]:
        if k and k in sem_map:
            br = str(sem_map[k].get("semantic_branch") or "")
            if br: return br
    stage = str(r.get("source_stage") or "")
    if "P4D" in stage: return "D15-P4D_FULL_REPLAY_CURRENT_INTEGRAL_BRANCH"
    if "P0" in stage or "P3" in stage or "P4B" in stage or "RG" in stage: return "D15-RG_REPAIR_FROM_SOURCE_SOFTLABEL_BRANCH"
    return "UNKNOWN_OR_MIXED_BRANCH"


def choose_stratified(records: Sequence[Mapping[str, Any]], split: str, limit: int, sem_map: Mapping[str, Mapping[str, str]], seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    import random
    rows = [dict(r) for r in records if str(r.get("split")) == split and not bool(r.get("is_flagged_probe"))]
    if limit <= 0 or len(rows) <= limit:
        return rows, [{"stage": "all_selected", "split": split, "count": len(rows)}]
    rng = random.Random(int(seed) + (17 if split == "train" else 31))
    proto_order = ["2C", "3C", "R2.5", "R3", "random_walk", "GEO", "UNKNOWN"]
    selected: List[Dict[str, Any]] = []; selected_ids = set(); audit: List[Dict[str, Any]] = []
    def add(row: Dict[str, Any], stage: str) -> None:
        uid = canonical_uid(row)
        if uid in selected_ids or len(selected) >= limit: return
        selected.append(row); selected_ids.add(uid)
        audit.append({"stage": stage, "split": split, "canonical_cell_uid": uid, "protocol": protocol_of(row), "semantic_branch": branch_of(row, sem_map)})
    for proto in proto_order:
        cands = [r for r in rows if canonical_uid(r) not in selected_ids and protocol_of(r) == proto]
        if cands: add(sorted(cands, key=lambda r: canonical_uid(r))[0], "one_per_protocol")
        if len(selected) >= limit: break
    for br in sorted({branch_of(r, sem_map) for r in rows}):
        if any(branch_of(r, sem_map) == br for r in selected): continue
        cands = [r for r in rows if canonical_uid(r) not in selected_ids and branch_of(r, sem_map) == br]
        if cands: add(sorted(cands, key=lambda r: canonical_uid(r))[0], "one_per_semantic_branch")
        if len(selected) >= limit: break
    remaining = [r for r in rows if canonical_uid(r) not in selected_ids]
    rng.shuffle(remaining)
    remaining = sorted(remaining, key=lambda r: (proto_order.index(protocol_of(r)) if protocol_of(r) in proto_order else 999, branch_of(r, sem_map), canonical_uid(r)))
    for r in remaining:
        add(r, "fill_balanced")
        if len(selected) >= limit: break
    return selected, audit


def make_progress_train_loop(out_dir: Path, progress_every: int, eval_every_override: int):
    from gv1.d17_g.g14_trainer import _group_balanced_loss  # type: ignore
    from gv1.d17_g.g13_trainer import _predict_np  # type: ignore
    from gv1.d17_g.g1_metrics import group_metrics  # type: ignore
    live_csv = out_dir / "D17_G7S1_training_progress_live.csv"
    live_jsonl = out_dir / "D17_G7S1_training_progress_live.jsonl"
    def _train_loop(model, loader, data, config, device, epochs, lr):
        model_cfg = dict(config.get("model", {}))
        opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(model_cfg.get("weight_decay", 2e-6)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(epochs)), eta_min=float(model_cfg.get("min_lr", 1e-5)))
        base_weights = dict(config.get("target_group_weights", {"theta_a": 1.5, "theta_c": 1.5, "cs_a": 1.0, "cs_c": 1.0, "phie": 16.0, "phis_c": 3.0}))
        phie_focus_epochs = int(config.get("phie_focus_epochs", 0)); phie_focus_multiplier = float(config.get("phie_focus_multiplier", 1.0)); non_phie_focus_scale = float(config.get("non_phie_focus_scale", 1.0))
        eval_every = int(eval_every_override or config.get("eval_every", 10)); prog_every = max(1, int(progress_every or config.get("progress_every", 1)))
        history: List[Dict[str, Any]] = []; best: Dict[str, Any] = {"epoch": 0, "score": -1e99, "fit_loss": float("inf"), "state_dict": None}
        start_wall = time.time()
        print(json.dumps({"event":"G7S1_TRAIN_START","epochs":int(epochs),"batches_per_epoch":int(len(loader)),"device":str(device),"live_csv":str(live_csv)}, ensure_ascii=False), flush=True)
        for ep in range(1, int(epochs)+1):
            ep_start = time.time(); weights = dict(base_weights); phase = "phie_focus" if phie_focus_epochs > 0 and ep <= phie_focus_epochs else "balanced"
            if phase == "phie_focus":
                weights["phie"] = float(weights.get("phie", 1.0)) * phie_focus_multiplier
                for k in ["theta_a","theta_c","cs_a","cs_c"]: weights[k] = float(weights.get(k, 1.0)) * non_phie_focus_scale
            model.train(); batch_losses: List[float] = []
            for xb, yb in loader:
                xb = xb.to(device=device, dtype=torch.float32); yb = yb.to(device=device, dtype=torch.float32)
                loss = _group_balanced_loss(model(xb), yb, data.base.target_slices, weights)
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(model_cfg.get("grad_clip_norm", 5.0))); opt.step()
                batch_losses.append(float(loss.detach().cpu()))
            scheduler.step(); fit_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
            elapsed = time.time()-start_wall; eta = elapsed/max(ep,1)*max(int(epochs)-ep,0)
            row: Dict[str, Any] = {"epoch":ep,"phase":phase,"fit_train_loss":fit_loss,"lr":float(opt.param_groups[0]["lr"]),"epoch_seconds":float(time.time()-ep_start),"elapsed_seconds":float(elapsed),"eta_seconds":float(eta)}
            do_eval = ep == 1 or ep == int(epochs) or ep % eval_every == 0
            if do_eval:
                model.eval()
                with torch.no_grad():
                    pred_fit = _predict_np(model, data.X_fit, data, device); fit_gm = group_metrics(data.Y_fit, pred_fit, data.base.target_slices)
                    row["fit_train_r2_mean"] = float(fit_gm["__aggregate__"]["r2_mean"]); row["fit_train_r2_min"] = float(fit_gm["__aggregate__"]["r2_min"]); row["fit_phie_r2"] = float(fit_gm.get("phie",{}).get("r2",float("nan")))
                    if getattr(data, "X_internal", np.zeros((0,0))).shape[0] > 0:
                        pred_int = _predict_np(model, data.X_internal, data, device); int_gm = group_metrics(data.Y_internal, pred_int, data.base.target_slices)
                        row["internal_heldout_r2_mean"] = float(int_gm["__aggregate__"]["r2_mean"]); row["internal_heldout_r2_min"] = float(int_gm["__aggregate__"]["r2_min"]); row["internal_phie_r2"] = float(int_gm.get("phie",{}).get("r2",float("nan")))
                    score = row["fit_train_r2_mean"] + 0.2*row["fit_train_r2_min"] + 0.3*row["fit_phie_r2"]
                    if np.isfinite(row.get("internal_heldout_r2_mean", float("nan"))): score += 0.7*row["internal_heldout_r2_mean"] + 0.2*row["internal_heldout_r2_min"] + 0.7*row["internal_phie_r2"]
                    score -= 0.005*fit_loss; row["selection_score"] = float(score) if np.isfinite(score) else float("nan")
                    if np.isfinite(score) and score > float(best.get("score", -1e99)):
                        best = {"epoch":ep,"score":float(score),"fit_loss":fit_loss,"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()}}; row["best_updated"] = True
                    else: row["best_updated"] = False
            history.append(row); write_csv(history, live_csv)
            with live_jsonl.open("a", encoding="utf-8") as f: f.write(json.dumps(row, ensure_ascii=False)+"\n")
            if ep == 1 or ep == int(epochs) or ep % prog_every == 0 or row.get("best_updated"):
                print(json.dumps({"event":"G7S1_EPOCH","epoch":ep,"epochs":int(epochs),"phase":phase,"loss":compact_float(fit_loss),"lr":compact_float(row["lr"]),"fit_r2_mean":compact_float(row.get("fit_train_r2_mean")),"fit_r2_min":compact_float(row.get("fit_train_r2_min")),"internal_r2_mean":compact_float(row.get("internal_heldout_r2_mean")),"internal_r2_min":compact_float(row.get("internal_heldout_r2_min")),"best_epoch":int(best.get("epoch",0)),"best_score":compact_float(best.get("score")),"elapsed":hhmmss(elapsed),"eta":hhmmss(eta)}, ensure_ascii=False), flush=True)
        print(json.dumps({"event":"G7S1_TRAIN_DONE","best_epoch":int(best.get("epoch",0)),"best_score":compact_float(best.get("score")),"elapsed":hhmmss(time.time()-start_wall),"live_csv":str(live_csv)}, ensure_ascii=False), flush=True)
        return best, history
    return _train_loop


def extract_s1_compact(summary: Mapping[str, Any]) -> Dict[str, Any]:
    fit = summary.get("fit_train_per_target_aggregate", {}) if isinstance(summary.get("fit_train_per_target_aggregate"), Mapping) else {}
    internal = summary.get("internal_heldout_per_target_aggregate", {}) if isinstance(summary.get("internal_heldout_per_target_aggregate"), Mapping) else {}
    val = summary.get("validation_report_only_per_target_aggregate", {}) if isinstance(summary.get("validation_report_only_per_target_aggregate"), Mapping) else {}
    return {"status": summary.get("status"), "best_epoch": summary.get("best_epoch"), "fit_train_mean_r2": fit.get("all_target_profile_r2_mean"), "fit_train_min_r2": fit.get("all_target_profile_r2_min"), "internal_heldout_mean_r2": internal.get("all_target_profile_r2_mean"), "internal_heldout_min_r2": internal.get("all_target_profile_r2_min"), "validation_mean_r2": val.get("all_target_profile_r2_mean"), "validation_min_r2": val.get("all_target_profile_r2_min"), "validation_phie_min_r2": val.get("phie_r2_min"), "worst_internal_target_profile": summary.get("worst_internal_target_profile"), "worst_validation_target_profile": summary.get("worst_validation_target_profile")}


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f: return json.load(f)


def build_temp_manifest(original: Mapping[str, Any], selected_train: Sequence[Mapping[str, Any]], selected_val: Sequence[Mapping[str, Any]], out_path: Path) -> None:
    d = dict(original); d["records"] = [dict(r) for r in selected_train] + [dict(r) for r in selected_val]
    d["counts"] = {"train": len(selected_train), "validation": len(selected_val)}
    d["source_manifest_hash_sha256"] = original.get("manifest_hash_sha256")
    d["manifest_hash_sha256"] = f"G7S1_TEMP_FROM_{original.get('manifest_hash_sha256', 'UNKNOWN')}"
    d["g7s1_temp_manifest_note"] = "Temporary stratified small-smoke manifest. It is not a replacement for the locked D17 split manifest."
    write_json(d, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="D17-G7-S1 small full-cycle smoke with real-time epoch progress")
    ap.add_argument("--config", required=True); ap.add_argument("--split_manifest", required=True); ap.add_argument("--g0_profile_semantics_csv", required=True); ap.add_argument("--s0_summary", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_profile_count", type=int, default=8); ap.add_argument("--validation_profile_count", type=int, default=2); ap.add_argument("--internal_heldout_count", type=int, default=2)
    ap.add_argument("--max_time_points", type=int, default=4096); ap.add_argument("--time_window_s", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=180); ap.add_argument("--lr", type=float, default=6e-4); ap.add_argument("--batch_size", type=int, default=2048); ap.add_argument("--device", default="auto")
    ap.add_argument("--progress_every", type=int, default=1); ap.add_argument("--eval_every", type=int, default=10); ap.add_argument("--allow_without_s0_pass", action="store_true")
    args = ap.parse_args()
    if int(args.max_time_points) <= 0: raise SystemExit("S1 smoke must use finite --max_time_points, e.g. 2048 or 4096. Do not use 0 for training.")
    if float(args.time_window_s) != 0.0: raise SystemExit("S1 must use --time_window_s 0.0 so sampling covers the full profile, not the first 40 ks.")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    s0 = read_json(args.s0_summary, default={}) or {}
    if not args.allow_without_s0_pass and (s0.get("status") != "PASS" or not bool(s0.get("s1_ready"))):
        raise SystemExit(f"S0 is not ready for S1. status={s0.get('status')} s1_ready={s0.get('s1_ready')} summary={args.s0_summary}")
    cfg = load_config(args.config); cfg["protocol"] = "D17-G7-S1_SMALL_FULLCYCLE_SMOKE"; cfg["seed"] = int(cfg.get("seed", 20260615)); cfg["internal_heldout_profile_count"] = int(args.internal_heldout_count); cfg["eval_every"] = int(args.eval_every); cfg["progress_every"] = int(args.progress_every)
    cfg["full_cycle_smoke_sampling"] = {"max_time_points_per_profile": int(args.max_time_points), "time_window_s": float(args.time_window_s), "purpose": "Small full-cycle smoke to test whether full-profile coverage sampling is worth scaling to S2."}
    manifest = read_json(args.split_manifest, default={}) or {}; records = list(manifest.get("records", [])); sem_map = load_semantics(args.g0_profile_semantics_csv)
    selected_train, train_audit = choose_stratified(records, "train", int(args.train_profile_count), sem_map, int(cfg.get("seed", 20260615)))
    selected_val, val_audit = choose_stratified(records, "validation", int(args.validation_profile_count), sem_map, int(cfg.get("seed", 20260615)))
    if not selected_train: raise SystemExit("No train profiles selected for S1")
    temp_manifest = out / "D17_G7S1_SELECTED_SMALL_SMOKE_SPLIT_MANIFEST.json"; build_temp_manifest(manifest, selected_train, selected_val, temp_manifest)
    write_csv(train_audit + val_audit, out / "D17_G7S1_SELECTED_PROFILE_AUDIT.csv"); write_json(cfg, out / "D17_G7S1_EFFECTIVE_CONFIG.json")
    print(json.dumps({"event":"G7S1_SELECTED_PROFILES","train_profile_count":len(selected_train),"validation_profile_count":len(selected_val),"train_profiles":[canonical_uid(r) for r in selected_train],"validation_profiles":[canonical_uid(r) for r in selected_val],"temp_manifest":str(temp_manifest)}, ensure_ascii=False, indent=2), flush=True)
    import gv1.d17_g.g2_trainer as g2_trainer  # type: ignore
    g2_trainer._train_loop = make_progress_train_loop(out, progress_every=int(args.progress_every), eval_every_override=int(args.eval_every))
    t0 = time.time()
    summary = g2_trainer.build_and_train_g2(split_manifest=str(temp_manifest), g0_profile_semantics_csv=args.g0_profile_semantics_csv, out_dir=str(out), config=cfg, train_profile_count=len(selected_train), validation_profile_count=len(selected_val), max_time_points=int(args.max_time_points), time_window_s=float(args.time_window_s), device_arg=args.device, epochs=int(args.epochs), lr=float(args.lr), batch_size=int(args.batch_size))
    s1 = dict(summary); s1["protocol"] = "D17-G7-S1_SMALL_FULLCYCLE_SMOKE"; s1["created_at_utc_g7s1_wrapper"] = utc_now(); s1["source_s0_summary"] = str(args.s0_summary); s1["source_s0_compact"] = {"status": s0.get("status"), "s1_ready": s0.get("s1_ready"), "coverage_gate": s0.get("coverage_gate")}; s1["training_wall_seconds_g7s1_wrapper"] = time.time()-t0
    s1["small_smoke_selected_train_profiles"] = [canonical_uid(r) for r in selected_train]; s1["small_smoke_selected_validation_profiles"] = [canonical_uid(r) for r in selected_val]; s1["temp_selected_manifest"] = str(temp_manifest)
    s1["purpose"] = "Small full-cycle coverage training smoke. It tests whether full-profile sampled training can improve dense selected-cycle behavior before any long S2 training."
    s1["policy"] = {"train_cell_softlabels_used_for_training": True, "validation_softlabels_report_only": True, "frozen_test_softlabels_used": False, "checkpoint_selection": "fit-train plus small protocol/branch-stratified train-internal heldout only; validation report-only is not used for checkpoint selection", "not_a_G6_or_full_allcycle_run": True, "not_a_S2_formal_run": True}
    compact = extract_s1_compact(s1); s1["compact_metrics"] = compact
    fit_ok = compact.get("fit_train_mean_r2") is not None and float(compact.get("fit_train_mean_r2") or -999) >= float(cfg.get("s1_fit_train_mean_r2_gate", 0.90))
    int_ok = compact.get("internal_heldout_mean_r2") is not None and float(compact.get("internal_heldout_mean_r2") or -999) >= float(cfg.get("s1_internal_mean_r2_gate", 0.70))
    val_ok = compact.get("validation_mean_r2") is not None and float(compact.get("validation_mean_r2") or -999) >= float(cfg.get("s1_validation_mean_r2_gate", 0.60))
    s1["selected_cycle_check_ready"] = bool(s1.get("status") == "PASS" and fit_ok and int_ok and val_ok); s1["s2_ready"] = False
    s1["recommendation"] = "RUN_G6F_SELECTED_CYCLE_DENSE_CHECK_BEFORE_S2" if s1["selected_cycle_check_ready"] else "DO_NOT_ENTER_S2_REVIEW_S1_TRAINING"
    s1["next_required_checks"] = ["Run G6F selected-cycle dense checks on Batch-2 battery-3 cycles 1-4 and 36-38 using this S1 candidate.", "Only consider G7-S2-mini if selected-cycle dense metrics improve substantially over G21/G4."]
    summary_path = out / "D17_G7S1_SMALL_FULLCYCLE_SMOKE_SUMMARY.json"; write_json(s1, summary_path); write_json(s1, out / "D17_G7S1_CANDIDATE_FOR_SELECTED_CYCLE_SUMMARY.json")
    for src_name, dst_name in {"D17_G2_training_history.csv":"D17_G7S1_training_history.csv","D17_G2_PER_TARGET_PROFILE_METRICS.csv":"D17_G7S1_PER_TARGET_PROFILE_METRICS.csv","D17_G2_PROFILE_METRICS.csv":"D17_G7S1_PROFILE_METRICS.csv","D17_G2_STRATIFIED_SPLIT_AUDIT.csv":"D17_G7S1_INTERNAL_SPLIT_AUDIT.csv"}.items():
        src = out / src_name; dst = out / dst_name
        if src.exists():
            try: dst.write_bytes(src.read_bytes())
            except Exception: pass
    print(json.dumps({"status":s1.get("status"),"selected_cycle_check_ready":s1.get("selected_cycle_check_ready"),"s2_ready":s1.get("s2_ready"),"recommendation":s1.get("recommendation"),"best_epoch":s1.get("best_epoch"),"compact_metrics":compact,"summary_json":str(summary_path),"candidate_summary_json":str(out/"D17_G7S1_CANDIDATE_FOR_SELECTED_CYCLE_SUMMARY.json"),"live_progress_csv":str(out/"D17_G7S1_training_progress_live.csv")}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
