#!/usr/bin/env python
"""Waterbirds ResNet18 runs from the CutClean paper (Tables 7-8).

The target classes are waterbirds vs landbirds and the private attribute is the
background, so P_threshold = 65 % as for the other two-class datasets.

Launch with no arguments:

    python experiments/run_waterbirds_resnet18.py

Per seed it runs:
  * a baseline run   -> gamma = 0, no pruning       ("Baseline" row of Table 8)
  * a full run       -> gamma grid + sparsity grid  (Table 7 + "CutClean" row of Table 8)

That is 3 seeds x 2 runs = 6 runs, executed sequentially. Re-launching skips the
runs that already produced a manifest.json.

Data are read from <CUTCLEAN_DATA>/waterbirds_unbiased
(train/val/test).

Note: waterbirds_dataloader.py keeps the images at their original resolution,
while the paper resizes every dataset to 224x224. Changing that would mean
touching the pipeline, so it is left as is; see experiments/README.md.

To change the epoch budget, the seeds or the switches (DRY_RUN, RUN_BASELINE,
WANDB_MODE), edit experiments/_paper_common.py.
"""

import sys

from _paper_common import run_campaign

CONFIGURATIONS = [
    {
        # Table 7 (gamma selection) and Table 8 (results).
        "key": "waterbirds",
        "dataset": "unbiasedWaterbirds",
        "t_ph": 0.65,     # P_threshold = 65 % for the two-class datasets
        "gamma_paper": 1, # gamma* selected in Table 7
    },
]

NODE = "nodemm06"   # SLURM node for this dataset (partition set in _paper_common.py)

if __name__ == "__main__":
    ok = run_campaign("CutClean ResNet18 - Waterbirds", CONFIGURATIONS, node=NODE)
    sys.exit(0 if ok else 1)
