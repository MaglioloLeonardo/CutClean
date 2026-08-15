#!/usr/bin/env python
"""Corrupted-CIFAR10 ResNet18 runs from the CutClean paper (Tables 5-6).

The private attribute is the image corruption, so there are ten private classes
and the paper lowers the privacy threshold to P_threshold = 20 %.

Launch with no arguments:

    python experiments/run_corrupted_cifar10_resnet18.py

Per seed it runs:
  * a baseline run   -> gamma = 0, no pruning       ("Baseline" row of Table 6)
  * a full run       -> gamma grid + sparsity grid  (Table 5 + "CutClean" row of Table 6)

That is 3 seeds x 2 runs = 6 runs, executed sequentially. Re-launching skips the
runs that already produced a manifest.json.

Data are read from <CUTCLEAN_DATA>/corrupted_cifar_unbiased
(train/valid/test); the loader maps the "val" split name onto the "valid"
folder on its own.

To change the epoch budget, the seeds or the switches (DRY_RUN, RUN_BASELINE,
WANDB_MODE), edit experiments/_paper_common.py.
"""

import sys

from _paper_common import run_campaign

CONFIGURATIONS = [
    {
        # Table 5 (gamma selection) and Table 6 (results).
        "key": "corrupted_cifar10",
        "dataset": "corruptedcifarunbiased",
        "t_ph": 0.20,        # P_threshold = 20 % for the ten-class synthetic dataset
        "gamma_paper": 1000, # gamma* selected in Table 5
    },
]

NODE = "nodemm03"   # SLURM node for this dataset (partition set in _paper_common.py)

if __name__ == "__main__":
    ok = run_campaign("CutClean ResNet18 - Corrupted-CIFAR10", CONFIGURATIONS,
                      node=NODE)
    sys.exit(0 if ok else 1)
