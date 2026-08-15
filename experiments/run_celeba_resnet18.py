#!/usr/bin/env python
"""CelebA ResNet18 runs from the CutClean paper.

Covers both target attributes: "blonde" (Tables 1-2) and "heavy make-up"
(Tables 3-4). The private attribute is gender in both cases.

Launch with no arguments:

    python experiments/run_celeba_resnet18.py

Per attribute and per seed it runs:
  * a baseline run   -> gamma = 0, no pruning       ("Baseline" table rows)
  * a full run       -> gamma grid + sparsity grid  (gamma table + "CutClean" rows)

That is 2 attributes x 3 seeds x 2 runs = 12 runs, executed sequentially.
Re-launching skips the runs that already produced a manifest.json.

To change the epoch budget, the seeds or the switches (DRY_RUN, RUN_BASELINE,
WANDB_MODE), edit experiments/_paper_common.py.

To force the paper's gamma* instead of re-selecting it, replace GAMMA_GRID in
_paper_common.py with the single value reported below.
"""

import sys

from _paper_common import run_campaign

CONFIGURATIONS = [
    {
        # Table 1 (gamma selection) and Table 2 (results).
        "key": "celeba_blonde",
        "dataset": "celeba-Blond_Hair-Male",
        "t_ph": 0.65,        # P_threshold = 65 % for the two-class datasets
        "gamma_paper": 1000, # gamma* selected in Table 1
    },
    {
        # Table 3 (gamma selection) and Table 4 (results).
        "key": "celeba_makeup",
        "dataset": "celeba-Heavy_Makeup-Male",
        "t_ph": 0.65,
        "gamma_paper": 5,    # gamma* selected in Table 3
    },
]

NODE = "nodemm02"   # SLURM node for this dataset (partition set in _paper_common.py)

if __name__ == "__main__":
    ok = run_campaign("CutClean ResNet18 - CelebA (blonde + heavy make-up)",
                      CONFIGURATIONS, node=NODE)
    sys.exit(0 if ok else 1)
