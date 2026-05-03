#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASSB myNN logging patch v2

Purpose
-------
Patch the current local util/myNN.py in place so that LogFin_*/log.csv records
loss components, especially reg_loss, without rewriting the whole myNN.py file.

What it changes
---------------
1. Backs up util/myNN.py.
2. Patches the SGD logTraining call to pass epoch_int/epoch_bound/epoch_data/epoch_reg.
3. Appends a small monkey-patch block that overrides myNN.prepareLog and
   myNN.logTraining to write a detailed log.csv header and rows.

This is intentionally minimal: it keeps the existing model, loss, optimizer,
checkpoint, and training-loop logic intact.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import shutil
import sys
from pathlib import Path

PATCH_MARKER = "# --- ASSB_DIAGNOSTIC_LOGGING_PATCH_V2 ---"


def _repo_root() -> Path:
    here = Path.cwd()
    if (here / "util" / "myNN.py").exists():
        return here
    if here.name == "util" and (here / "myNN.py").exists():
        return here.parent
    raise SystemExit("ERROR: run this script from the project root or util folder.")


def _patch_sgd_logtraining_call(text: str) -> tuple[str, bool]:
    # Patch the exact current GitHub/local SGD summary logging call.
    old = "bestLoss = self.logTraining(epoch + 1, epoch_loss, bestLoss, mse_unweighted=final_unweighted)"
    new = (
        "bestLoss = self.logTraining(\n"
        "                    epoch + 1,\n"
        "                    epoch_loss,\n"
        "                    bestLoss,\n"
        "                    mse_unweighted=final_unweighted,\n"
        "                    int_loss=epoch_int,\n"
        "                    bound_loss=epoch_bound,\n"
        "                    data_loss=epoch_data,\n"
        "                    reg_loss=epoch_reg,\n"
        "                    stage=\"SGD\",\n"
        "                )"
    )
    if old in text:
        return text.replace(old, new), True

    # More tolerant single-line fallback, but do not touch already-patched calls.
    pat = re.compile(
        r"bestLoss\s*=\s*self\.logTraining\(\s*epoch\s*\+\s*1\s*,\s*epoch_loss\s*,\s*bestLoss\s*,\s*mse_unweighted\s*=\s*final_unweighted\s*\)"
    )
    if pat.search(text):
        return pat.sub(new, text), True
    return text, False


MONKEY_PATCH = r'''

# --- ASSB_DIAGNOSTIC_LOGGING_PATCH_V2 ---
# Appended by apply_myNN_logging_patch_v2.py.
# Purpose: make reg_loss / int_loss / bound_loss / data_loss visible in LogFin_*/log.csv.

def _assb_float_for_log_v2(value, default=float("nan")):
    try:
        if value is None:
            return float(default)
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().reshape(-1)[0])
        return float(value)
    except Exception:
        return float(default)


def _assb_jsonable_v2(value):
    try:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [_assb_jsonable_v2(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _assb_jsonable_v2(v) for k, v in value.items()}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
    except Exception:
        return str(value)


def _assb_prepareLog_v2(self):
    os.makedirs(self.logLossFolder, exist_ok=True)
    os.makedirs(self.modelFolder, exist_ok=True)
    self.total_step = int(getattr(self, "total_step", 0))

    # Detailed epoch-level log. This intentionally supersedes the old header
    # "epoch;step;mseloss" that hid regularization diagnostics.
    header = (
        "epoch;step;stage;loss;unweighted_loss;int_loss;bound_loss;data_loss;reg_loss;"
        "activeInt;activeBound;activeData;activeReg;"
        "alpha0;alpha1;alpha2;alpha3;"
        "batch_size_int;batch_size_bound;batch_size_data;batch_size_reg;"
        "w_cs_a_mass_reg_effective;w_cs_c_mass_reg_effective\n"
    )
    with open(os.path.join(self.logLossFolder, "log.csv"), "w", encoding="utf-8") as f:
        f.write(header)

    # Keep a separate detailed file with the same rows for convenience.
    with open(os.path.join(self.logLossFolder, "log_detailed.csv"), "w", encoding="utf-8") as f:
        f.write(header)

    diag = {
        "activeInt": bool(getattr(self, "activeInt", False)),
        "activeBound": bool(getattr(self, "activeBound", False)),
        "activeData": bool(getattr(self, "activeData", False)),
        "activeReg": bool(getattr(self, "activeReg", False)),
        "alpha": _assb_jsonable_v2(getattr(self, "alpha", None)),
        "batch_size_int": int(getattr(self, "batch_size_int", -1)),
        "batch_size_bound": int(getattr(self, "batch_size_bound", -1)),
        "batch_size_data": int(getattr(self, "batch_size_data", -1)),
        "batch_size_reg": int(getattr(self, "batch_size_reg", -1)),
        "regTerms_rescale": _assb_jsonable_v2(getattr(self, "regTerms_rescale", None)),
        "regTerms_rescale_unweighted": _assb_jsonable_v2(getattr(self, "regTerms_rescale_unweighted", None)),
        "w_cs_a_mass_reg_effective": _assb_jsonable_v2(self.params.get("w_cs_a_mass_reg_effective", self.params.get("w_cs_a_mass_reg", None))),
        "w_cs_c_mass_reg_effective": _assb_jsonable_v2(self.params.get("w_cs_c_mass_reg_effective", self.params.get("w_cs_c_mass_reg", None))),
        "mass_reg_n_quad": _assb_jsonable_v2(self.params.get("mass_reg_n_quad", None)),
        "train_summary_json": _assb_jsonable_v2(self.params.get("train_summary_json", self.params.get("soft_label_summary_path", None))),
    }
    try:
        with open(os.path.join(self.modelFolder, "assb_training_diagnostics.json"), "w", encoding="utf-8") as f:
            json.dump(diag, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"WARNING: could not write assb_training_diagnostics.json: {exc}")


def _assb_logTraining_v2(
    self,
    epoch,
    mseloss,
    bestLoss,
    mse_unweighted=None,
    int_loss=None,
    bound_loss=None,
    data_loss=None,
    reg_loss=None,
    stage=None,
):
    loss_f = _assb_float_for_log_v2(mseloss)
    unweighted_f = _assb_float_for_log_v2(mse_unweighted)
    int_f = _assb_float_for_log_v2(int_loss)
    bound_f = _assb_float_for_log_v2(bound_loss)
    data_f = _assb_float_for_log_v2(data_loss)
    reg_f = _assb_float_for_log_v2(reg_loss)
    stage = str(stage or getattr(self, "current_stage", "NA"))

    alpha_vals = list(getattr(self, "alpha", [float("nan")] * 4))
    while len(alpha_vals) < 4:
        alpha_vals.append(float("nan"))

    row = (
        f"{int(epoch)};{int(getattr(self, 'total_step', -1))};{stage};"
        f"{loss_f:.16g};{unweighted_f:.16g};{int_f:.16g};{bound_f:.16g};{data_f:.16g};{reg_f:.16g};"
        f"{bool(getattr(self, 'activeInt', False))};{bool(getattr(self, 'activeBound', False))};"
        f"{bool(getattr(self, 'activeData', False))};{bool(getattr(self, 'activeReg', False))};"
        f"{_assb_float_for_log_v2(alpha_vals[0]):.16g};{_assb_float_for_log_v2(alpha_vals[1]):.16g};"
        f"{_assb_float_for_log_v2(alpha_vals[2]):.16g};{_assb_float_for_log_v2(alpha_vals[3]):.16g};"
        f"{int(getattr(self, 'batch_size_int', -1))};{int(getattr(self, 'batch_size_bound', -1))};"
        f"{int(getattr(self, 'batch_size_data', -1))};{int(getattr(self, 'batch_size_reg', -1))};"
        f"{_assb_float_for_log_v2(self.params.get('w_cs_a_mass_reg_effective', self.params.get('w_cs_a_mass_reg', float('nan')))):.16g};"
        f"{_assb_float_for_log_v2(self.params.get('w_cs_c_mass_reg_effective', self.params.get('w_cs_c_mass_reg', float('nan')))):.16g}\n"
    )
    for fname in ("log.csv", "log_detailed.csv"):
        with open(os.path.join(self.logLossFolder, fname), "a", encoding="utf-8") as f:
            f.write(row)

    # Preserve simple checkpoint semantics without touching optimizer logic.
    improved = bestLoss is None or loss_f < float(bestLoss)
    if improved:
        bestLoss = loss_f
        try:
            safe_save_state_dict(self.model, os.path.join(self.modelFolder, "best.pt"))
        except Exception as exc:
            print(f"WARNING: could not save best.pt: {exc}")
        try:
            cfg = dict(getattr(self, "configDict", {}))
            cfg["alpha"] = _assb_jsonable_v2(getattr(self, "alpha", None))
            cfg["activeReg"] = bool(getattr(self, "activeReg", False))
            cfg["BATCH_SIZE_REG"] = int(getattr(self, "batch_size_reg", -1))
            cfg["regTerms_rescale"] = _assb_jsonable_v2(getattr(self, "regTerms_rescale", None))
            cfg["regTerms_rescale_unweighted"] = _assb_jsonable_v2(getattr(self, "regTerms_rescale_unweighted", None))
            cfg["w_cs_a_mass_reg_effective"] = _assb_jsonable_v2(self.params.get("w_cs_a_mass_reg_effective", self.params.get("w_cs_a_mass_reg", None)))
            cfg["w_cs_c_mass_reg_effective"] = _assb_jsonable_v2(self.params.get("w_cs_c_mass_reg_effective", self.params.get("w_cs_c_mass_reg", None)))
            with open(os.path.join(self.modelFolder, "config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"WARNING: could not write config.json: {exc}")

    try:
        safe_save_state_dict(self.model, os.path.join(self.modelFolder, "last.pt"))
    except Exception:
        pass
    return bestLoss


# Override methods after class definition. Existing training loop calls these names.
myNN.prepareLog = _assb_prepareLog_v2
myNN.logTraining = _assb_logTraining_v2
# --- END ASSB_DIAGNOSTIC_LOGGING_PATCH_V2 ---
'''


def main() -> int:
    root = _repo_root()
    path = root / "util" / "myNN.py"
    if not path.exists():
        raise SystemExit(f"ERROR: file not found: {path}")

    text = path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        print("INFO: myNN.py already contains ASSB diagnostic logging patch v2.")
        return 0

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak_assb_diag_{stamp}")
    shutil.copy2(path, backup)

    text2, patched_call = _patch_sgd_logtraining_call(text)
    if not patched_call:
        print("WARNING: did not find the exact SGD logTraining call. The monkey-patched logTraining will still work, but log.csv may not receive component losses unless train() passes them.")

    text2 = text2.rstrip() + MONKEY_PATCH + "\n"
    path.write_text(text2, encoding="utf-8", newline="\n")

    print(f"Patched: {path}")
    print(f"Backup : {backup}")
    print(f"SGD logTraining call patched: {patched_call}")
    print("Next check: Select-String .\\util\\myNN.py -Pattern 'ASSB_DIAGNOSTIC_LOGGING_PATCH_V2|int_loss=epoch_int|reg_loss=epoch_reg'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
