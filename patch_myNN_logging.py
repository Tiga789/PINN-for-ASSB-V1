#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch util/myNN.py to log loss components and regularization diagnostics.

Run from the PINN-for-ASSB-V1 repository root:
    python patch_myNN_logging.py

It creates util/myNN.py.bak_assb_logging before modifying util/myNN.py.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MYNN = ROOT / "util" / "myNN.py"
if not MYNN.exists():
    raise SystemExit(f"Cannot find {MYNN}. Put this script in the repository root and rerun.")

text = MYNN.read_text(encoding="utf-8")
backup = MYNN.with_suffix(".py.bak_assb_logging")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

# 1. Header: expose component losses.
text = text.replace(
    'f.write("epoch;step;mseloss\\n")',
    'f.write("epoch;step;loss;unweighted_loss;int_loss;bound_loss;data_loss;reg_loss\\n")',
)

# 2. Replace logTraining with a component-aware version. Keep checkpoint logic identical.
pattern = re.compile(
    r"    def logTraining\(self, epoch, mse, bestLoss, mse_unweighted=None\):\n"
    r".*?"
    r"        return bestLoss\n"
    r"    def logLosses\(",
    re.DOTALL,
)
new_func = '''    def logTraining(
        self,
        epoch,
        mse,
        bestLoss,
        mse_unweighted=None,
        int_loss=None,
        bound_loss=None,
        data_loss=None,
        reg_loss=None,
    ):
        # Keep the original checkpoint behavior, but make log.csv diagnostic.
        # This is critical for ASSB debugging because alpha[3]/BATCH_SIZE_REG
        # can be nonzero while the scalar mseloss alone hides whether rL is active.
        unweighted = float("nan") if mse_unweighted is None else float(mse_unweighted)
        int_v = float("nan") if int_loss is None else float(int_loss)
        bound_v = float("nan") if bound_loss is None else float(bound_loss)
        data_v = float("nan") if data_loss is None else float(data_loss)
        reg_v = float("nan") if reg_loss is None else float(reg_loss)
        with open(os.path.join(self.logLossFolder, "log.csv"), "a+", encoding="utf-8") as f:
            f.write(
                f"{int(epoch)};{int(epoch * self.n_batch)};{float(mse)};{unweighted};"
                f"{int_v};{bound_v};{data_v};{reg_v}\\n"
            )
        epochLoss = float(mse_unweighted if mse_unweighted is not None else mse)
        safe_save_state_dict(self.model, os.path.join(self.modelFolder, "last.pt"))
        if self.current_stage.upper() == "SGD":
            safe_save_state_dict(self.model, os.path.join(self.modelFolder, "lastSGD.pt"))
        else:
            safe_save_state_dict(self.model, os.path.join(self.modelFolder, "lastLBFGS.pt"))
        if bestLoss is None or epochLoss < float(bestLoss):
            bestLoss = epochLoss
            safe_save_state_dict(self.model, os.path.join(self.modelFolder, "best.pt"))
        return bestLoss
    def logLosses('''
text, n = pattern.subn(new_func, text, count=1)
if n != 1:
    raise SystemExit("Could not patch logTraining(). Please upload util/myNN.py for a manual patch.")

# 3. Pass epoch component losses from SGD and LBFGS calls.
old = 'bestLoss = self.logTraining(epoch + 1, epoch_loss, bestLoss, mse_unweighted=final_unweighted)'
new = ('bestLoss = self.logTraining(epoch + 1, epoch_loss, bestLoss, '
       'mse_unweighted=final_unweighted, int_loss=epoch_int, bound_loss=epoch_bound, '
       'data_loss=epoch_data, reg_loss=epoch_reg)')
if old in text:
    text = text.replace(old, new)
else:
    print("WARNING: SGD logTraining call pattern not found; leaving it unchanged.")

old = 'bestLoss = self.logTraining(self.nEpochs + epoch + 1, epoch_loss, bestLoss, mse_unweighted=final_unweighted)'
new = ('bestLoss = self.logTraining(self.nEpochs + epoch + 1, epoch_loss, bestLoss, '
       'mse_unweighted=final_unweighted, int_loss=epoch_int, bound_loss=epoch_bound, '
       'data_loss=epoch_data, reg_loss=epoch_reg)')
if old in text:
    text = text.replace(old, new)
else:
    print("WARNING: LBFGS logTraining call pattern not found; leaving it unchanged.")

MYNN.write_text(text, encoding="utf-8")
print(f"Patched {MYNN}")
print(f"Backup saved to {backup}")
