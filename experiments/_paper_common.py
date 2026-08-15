"""Shared configuration and launch machinery for the ResNet18 paper runs.

This module is not meant to be executed. Launch one of the three entry points
instead:

    python experiments/run_celeba_resnet18.py
    python experiments/run_corrupted_cifar10_resnet18.py
    python experiments/run_waterbirds_resnet18.py

Every value below comes from "CutClean: Neural Network Pruning for
Privacy-Preserving Inference", Sec. 4.1 (Setup) and the result tables. Each run
is executed as a separate `python src/run_cutclean.py ...` subprocess, so that
seeding, GPU memory and the Weights & Biases run are fresh every time.
"""

import glob
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Paths on this machine
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(REPO_ROOT, "src", "run_cutclean.py")

DATAPATH = os.environ.get("CUTCLEAN_DATA", os.path.join(REPO_ROOT, "data")).rstrip(os.sep) + os.sep
CIFAR_ROOT = os.path.join(DATAPATH, "corrupted_cifar_unbiased")
WATERBIRDS_ROOT = os.path.join(DATAPATH, "waterbirds_unbiased")

LOG_DIR = os.path.join(REPO_ROOT, "logs")

# ---------------------------------------------------------------------------
# Behaviour switches - edit here, the entry points take no arguments
# ---------------------------------------------------------------------------
DRY_RUN = False          # True: print the commands without running them
SKIP_COMPLETED = True    # skip runs whose manifest.json already exists
RUN_BASELINE = True      # also run gamma=0 unpruned (the "Baseline" table rows)
WANDB_MODE = "online"    # "online", "offline" or "disabled"
PYTHON = None            # None: use the interpreter that launched the script,
                         # falling back to the "cutclean" conda env if it has no torch.
                         # Set an absolute path to force a specific interpreter.

# ---------------------------------------------------------------------------
# SLURM submission. With USE_SLURM the script submits one sbatch job per run,
# pinned to the node passed by the entry point, and returns immediately.
# Set USE_SLURM = False to run everything locally and sequentially instead.
# ---------------------------------------------------------------------------
USE_SLURM = True
PARTITION = "mm"
SLURM_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "cutclean_resnet18.slurm")
SLURM_GPUS = 1
SLURM_CPUS = 8
SLURM_MEM = "32G"
SLURM_TIME = "7-00:00:00"   # partition mm has no time limit; this is a safety bound

# ---------------------------------------------------------------------------
# Paper hyper-parameters (Sec. 4.1, "Training details")
#   SGD, momentum 0.9, weight decay 1e-4, batch size 124, initial lr 1e-2,
#   400 MI pre-training epochs, 10 fine-tuning epochs after pruning.
#   ResNet18 starts from ImageNet-1k weights and each residual stage is a block.
#   "Results are averaged across three runs" (Fig. 2 caption).
# ---------------------------------------------------------------------------
MODEL = "resnet18"
BATCH_SIZE = 124
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001

E_TASK = 0    # 0 = start the MI phase straight from ImageNet weights, as in the paper
E_PRE = 400   # MI pre-training epochs, per gamma
E_FT = 10     # MI fine-tuning epochs after each pruning
E_PH = 100    # privacy-head attack epochs; not fixed by the paper, repository default.
              # This is the cheapest knob if you need to shorten a campaign.

SEEDS = (0,)   # the paper averages over three runs: use (0, 1, 2) for the full campaign

# gamma grid shared by Tables 1, 3, 5 and 7.
GAMMA_GRID = (0, 0.01, 0.1, 1, 5, 10, 50, 100, 500, 1000)

# Fixed sparsity grid over s in [0, 1] (Sec. 3.2), step 0.05: it covers every
# sparsity level reported in the paper (5, 10, 20, 30, 40, 50, 60, 70, 95 %).
SPARSITY_GRID = tuple(round(0.05 * i, 2) for i in range(20))


def _fmt(value):
    """Render a number for the comma-separated CLI lists (1000 not 1000.0)."""
    number = float(value)
    return str(int(number)) if number == int(number) else repr(number)


def _joined(values):
    return ",".join(_fmt(v) for v in values)


def _has_torch(interpreter):
    probe = subprocess.run([interpreter, "-c", "import torch"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return probe.returncode == 0


def _resolve_interpreter():
    """Pick an interpreter that can actually import torch, and say which one."""
    if PYTHON:
        if not _has_torch(PYTHON):
            sys.exit(f"PYTHON is set to {PYTHON}, but that interpreter cannot import torch.")
        return PYTHON

    if _has_torch(sys.executable):
        return sys.executable

    fallback = os.path.expanduser("~/miniconda3/envs/cutclean/bin/python")
    if os.path.exists(fallback) and _has_torch(fallback):
        print(f"[preflight] {sys.executable} has no torch; using {fallback} instead.\n"
              f"[preflight] Activate the environment first ('conda activate cutclean') "
              f"to silence this.", flush=True)
        return fallback

    sys.exit("No interpreter with torch was found. Activate the training environment "
             "(e.g. 'conda activate cutclean') and launch this script again.")


def _preflight(configurations):
    problems = []
    if not os.path.isfile(DRIVER):
        problems.append(f"driver not found: {DRIVER}")
    if not os.path.isdir(DATAPATH):
        problems.append(f"datapath not found: {DATAPATH}")

    for config in configurations:
        dataset = config["dataset"]
        if dataset.startswith("celeba"):
            root = os.path.join(DATAPATH, "celeba")
        elif dataset.startswith("corruptedcifarunbiased"):
            root = CIFAR_ROOT
        elif dataset.startswith("unbiasedWaterbirds"):
            root = WATERBIRDS_ROOT
        else:
            continue
        if not os.path.isdir(root):
            problems.append(f"data for '{dataset}' not found: {root}")

    if problems:
        sys.exit("Preflight failed:\n  - " + "\n  - ".join(problems))


def _already_done(project):
    pattern = os.path.join(REPO_ROOT, "models", project, "*", "manifest.json")
    return sorted(glob.glob(pattern))


def _driver_args(project, dataset, gamma_list, sparsity_list, t_ph, seed):
    return [
        DRIVER,
        "--dataset", dataset,
        "--datapath", DATAPATH,
        "--corruptedcifarunbiased_root", CIFAR_ROOT,
        "--waterbirds_root", WATERBIRDS_ROOT,
        "--projectName", project,
        "--model", MODEL,
        "--seed", str(seed),
        "--gamma_list", _joined(gamma_list),
        "--sparsity_list", _joined(sparsity_list),
        "--t_ph", str(t_ph),
        "--e_task", str(E_TASK),
        "--e_pre", str(E_PRE),
        "--e_ft", str(E_FT),
        "--e_ph", str(E_PH),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--mom_sgd", str(MOMENTUM),
        "--wd", str(WEIGHT_DECAY),
    ]


def _launch_local(interpreter, project, args):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{project}.log")
    env = dict(os.environ, WANDB_MODE=WANDB_MODE)

    print(f"    log  -> {log_path}", flush=True)
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.run([interpreter] + args, cwd=REPO_ROOT, env=env,
                                 stdout=log_file, stderr=subprocess.STDOUT)
    minutes = (time.time() - started) / 60.0
    ok = process.returncode == 0
    print(f"    {'OK' if ok else 'FAILED (exit %d)' % process.returncode}"
          f" in {minutes:.1f} min", flush=True)
    return ok


def _sbatch_command(project, args, node):
    os.makedirs(LOG_DIR, exist_ok=True)
    return [
        "sbatch",
        f"--job-name={project}",
        f"--partition={PARTITION}",
        f"--nodelist={node}",
        f"--gres=gpu:{SLURM_GPUS}",
        f"--cpus-per-task={SLURM_CPUS}",
        f"--mem={SLURM_MEM}",
        f"--time={SLURM_TIME}",
        f"--chdir={REPO_ROOT}",
        f"--output={os.path.join(LOG_DIR, project + '.out')}",
        f"--error={os.path.join(LOG_DIR, project + '.err')}",
        f"--export=ALL,WANDB_MODE={WANDB_MODE}",
        SLURM_TEMPLATE,
    ] + args


def _submit_slurm(project, args, node):
    command = _sbatch_command(project, args, node)
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    SUBMIT FAILED: {result.stderr.strip()}", flush=True)
        return None
    job_id = result.stdout.strip().split()[-1]
    print(f"    submitted job {job_id} on {node}  "
          f"(log: logs/{project}.out)", flush=True)
    return job_id


def run_campaign(title, configurations, node=None):
    """Launch every (configuration, seed) pair.

    With USE_SLURM each run becomes an sbatch job pinned to `node`, and this
    function returns as soon as everything is queued. Otherwise the runs are
    executed locally, one after the other.

    `configurations` is a sequence of dicts with the keys:
        key      short name used in the project/log name
        dataset  value for the driver's --dataset
        t_ph     P_threshold on the last privacy head, as a fraction
        gamma_paper  gamma* reported in the paper (logged for reference only;
                     the pipeline re-selects gamma on its own)
    """
    _preflight(configurations)
    if USE_SLURM and not DRY_RUN and node is None:
        sys.exit("USE_SLURM is on but this entry point passed no node.")
    interpreter = None
    if not USE_SLURM:
        interpreter = _resolve_interpreter() if not DRY_RUN else sys.executable

    planned = []
    for config in configurations:
        for seed in SEEDS:
            if RUN_BASELINE:
                planned.append((
                    f"paper_{config['key']}_baseline_seed{seed}",
                    config, seed, (0,), (0.0,), "baseline (gamma=0, unpruned)",
                ))
            planned.append((
                f"paper_{config['key']}_seed{seed}",
                config, seed, GAMMA_GRID, SPARSITY_GRID, "gamma selection + sparsity sweep",
            ))

    print("=" * 74)
    print(f" {title}")
    print(f" runs         : {len(planned)}")
    print(f" seeds        : {', '.join(str(s) for s in SEEDS)}")
    print(f" epochs       : MI pre-train {E_PRE} | fine-tune {E_FT} | PH attack {E_PH}")
    print(f" data         : {DATAPATH}")
    if USE_SLURM:
        print(f" slurm        : partition {PARTITION}, node {node},"
              f" {SLURM_GPUS} GPU, {SLURM_CPUS} cpus, {SLURM_MEM}")
    else:
        print(f" python       : {interpreter}")
    print(f" results in   : {os.path.join(REPO_ROOT, 'models')}")
    print(f" wandb        : {WANDB_MODE}")
    if DRY_RUN:
        print(" DRY RUN      : nothing will be executed")
    print("=" * 74)

    done, failed, skipped = [], [], []
    try:
        for index, (project, config, seed, gammas, sparsities, what) in enumerate(planned, 1):
            print(f"\n[{index}/{len(planned)}] {project}")
            print(f"    {config['dataset']}  |  {what}  |  T_PH={config['t_ph']}"
                  f"  |  paper gamma*={config['gamma_paper']}")

            args = _driver_args(project, config["dataset"], gammas,
                                sparsities, config["t_ph"], seed)

            if DRY_RUN:
                preview = (_sbatch_command(project, args, node or "<node>")
                           if USE_SLURM else [interpreter] + args)
                print("    " + " ".join(preview))
                continue

            existing = _already_done(project) if SKIP_COMPLETED else []
            if existing:
                print(f"    already complete -> {existing[-1]}")
                skipped.append(project)
                continue

            if USE_SLURM:
                (done if _submit_slurm(project, args, node) else failed).append(project)
            else:
                (done if _launch_local(interpreter, project, args) else failed).append(project)
    except KeyboardInterrupt:
        print("\nInterrupted. Completed runs keep their results; re-launching "
              "this script resumes from where it stopped.")

    if not DRY_RUN:
        print("\n" + "=" * 74)
        verb = "submitted" if USE_SLURM else "completed"
        print(f" {verb} {len(done)} | skipped {len(skipped)} | failed {len(failed)}")
        for project in failed:
            print(f"   FAILED  {project}  (see {os.path.join(LOG_DIR, project + '.log')})")
        if USE_SLURM:
            print(" follow with: squeue -u $USER   |   tail -f logs/<projectName>.out")
        print(" per-run outputs: models/<projectName>/<run>/")
        print("   stats/gamma_selection.csv        gamma vs last-PH validation accuracy")
        print("   stats/results_sparsity_sweep.csv sparsity vs target/private accuracy")
        print("   manifest.json + pretrained_model.pth / mi_model.pth / pruned_model.pth")
        print("=" * 74)

    return not failed
