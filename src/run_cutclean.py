"""CutClean — Algorithm 1 driver (supplementary material pseudocode).

Implements the full pseudoalgorithm on top of the existing code base:

  Phase 0  Task pretraining        : train f on the task only (NO MI). This is
                                     the "pretrained model" (before the MI phase).
  Phase 1  GammaSelection          : for each gamma in --gamma_list, run
                                     onlineBatchMITraining (e_pre epochs) from the
                                     SAME task-pretrained f, freeze f, retrain the
                                     privacy heads, measure the last head's val
                                     accuracy; gamma* = argmin.
  Phase 2  Sparsity sweep          : for each s in --sparsity_list, prune the
                                     gamma* MI backbone at global sparsity s
                                     (L1-per-channel ranking, same as the original
                                     scripts), fine-tune e_ft epochs with online
                                     batch MI, freeze, retrain privacy heads, and
                                     evaluate target/private accuracy on
                                     train/val/test.
  Phase 3  Selection               : C_priv = {s : last-head val acc < T_PH};
                                     pick max target val acc in C_priv, else the
                                     model with the lowest last-head val acc.

Everything is saved inside a timestamped run directory under --save_path:

  pretrained_model.pth   task-trained model, before the MI phase
  mi_model.pth           unpruned model minimized with MI (gamma*)
  pruned_model.pth       final pruned model selected by Phase 3
  mi_gamma_<g>.pth       per-gamma MI backbones (Phase 1)
  pruned_s_<s>.pth       per-sparsity pruned backbones (Phase 2, masks folded)
  <tag>_ph<i>_<model>.pth  privacy-head checkpoints from each attack retraining
  stats/gamma_selection.csv   gamma, last-head val acc, target val acc
  stats/results_sparsity_sweep.csv  per-s sparsity + target/private accuracies
  stats/run_summary.txt       human-readable log
  manifest.json               paths of the three requested models + gamma*, s*

Example:
  python run_cutclean.py \
      --dataset celeba-Blond_Hair-Male --datapath <CUTCLEAN_DATA>/ \
      --projectName pseudocode_run \
      --gamma_list 0.1,0.3,0.5,1.0 \
      --sparsity_list 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
      --e_pre 100 --e_ft 10 --t_ph 0.65

Optional restarts:
  --reload_task_model PATH              skip Phase 0
  --reload_mi_model PATH --gamma_star G skip Phases 0-1
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import wandb

# ----------------------------------------------------------------------------
# Driver-specific CLI (stripped from sys.argv BEFORE config.parse_args runs).
# ----------------------------------------------------------------------------

def _parse_driver_args():
    # allow_abbrev=False: without it argparse would treat the project's own
    # --gamma flag as an ambiguous abbreviation of --gamma_list/--gamma_star.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--gamma_list", type=str, required=True,
                        help="Comma-separated candidate gammas (Gamma_list); each is applied uniformly to all privacy heads")
    parser.add_argument("--sparsity_list", type=str, required=True,
                        help="Comma-separated sparsities in [0,1] (S_list)")
    parser.add_argument("--t_ph", type=float, default=0.65,
                        help="T_PH threshold on last privacy-head VAL accuracy (fraction in [0,1])")
    parser.add_argument("--e_task", type=int, default=None,
                        help="Task-only pretraining epochs (Phase 0). Default: --nb_epochs")
    parser.add_argument("--e_pre", type=int, default=None,
                        help="MI pretraining epochs per gamma (Phase 1). Default: --nb_epochs")
    parser.add_argument("--e_ft", type=int, default=10,
                        help="Post-pruning MI fine-tuning epochs (Phase 2)")
    parser.add_argument("--e_ph", type=int, default=None,
                        help="Privacy-head retraining epochs for the attack. Default: --nb_epochs_ph[0]")
    parser.add_argument("--reload_task_model", type=str, default=None,
                        help="Checkpoint of the task-pretrained model; skips Phase 0")
    parser.add_argument("--reload_mi_model", type=str, default=None,
                        help="Checkpoint of the MI-minimized model; skips Phases 0-1 (requires --gamma_star)")
    parser.add_argument("--gamma_star", type=float, default=None,
                        help="Gamma of --reload_mi_model (required with it)")
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print("=== run_cutclean.py driver options ===")
        parser.print_help()
        print("\n=== options forwarded to config.parse_args ===")
        sys.argv = [sys.argv[0], "--help"]
        from config import parse_args
        parse_args()  # prints the project options and exits

    driver_args, remaining = parser.parse_known_args()

    if driver_args.reload_mi_model is not None and driver_args.gamma_star is None:
        parser.error("--reload_mi_model requires --gamma_star")

    # Hand the remaining argv back to the project's parser, injecting the
    # arguments it marks as required but that this driver supersedes.
    def _has_flag(flag):
        return any(a == flag or a.startswith(flag + "=") for a in remaining)

    sys.argv = [sys.argv[0]] + remaining
    if not _has_flag("--gamma"):
        sys.argv += ["--gamma", "0,0,0,0"]
    if not _has_flag("--pruning_retraining"):
        sys.argv += ["--pruning_retraining", "retrain"]

    from config import parse_args  # imported here so --help shows its options too
    args = parse_args()

    driver_args.gamma_list = [float(x) for x in driver_args.gamma_list.split(",")]
    driver_args.sparsity_list = [float(x) for x in driver_args.sparsity_list.split(",")]
    if driver_args.t_ph > 1.0:
        driver_args.t_ph /= 100.0
    for s in driver_args.sparsity_list:
        if not (0.0 <= s <= 1.0):
            raise ValueError(f"Sparsity {s} outside [0,1]")
    return args, driver_args


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _acc_to_frac(a):
    return a / 100.0 if a > 1.0 else a


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _as_list(value, dtype, length):
    if isinstance(value, list):
        return [dtype(v) for v in value]
    return [dtype(value)] * length


def _warm_up_heads(model, phs, dls, args):
    """Run one tiny real batch through the model and every privacy head.

    HeadStructure builds its Linear from a 224x224 dummy probe and lazily
    REBUILDS it on the first real forward if the dataset's feature size differs
    (irene/core.py _ensure_linear_matches). Settling the heads here, before any
    head checkpoint is loaded and before any optimizer captures the head
    parameters, prevents (a) strict-load shape mismatches and (b) optimizers
    silently pointing at discarded parameters.
    """
    model.eval()
    with torch.no_grad():
        data, _, _ = next(iter(dls["train"]))
        data = data[:2].to(args.device)
        if args.model == "vit":
            _ = model(data).logits
        else:
            _ = model(data)
        for ph in phs:
            _ = ph()


def fresh_model_and_phs(args, cc, train_mod, dls, state_dict=None, ph_paths=None):
    """Build a brand-new model (fresh forward hooks) + privacy heads.

    Rebuilding the model for every gamma/sparsity iteration avoids accumulating
    forward hooks on a shared backbone (irene.Hook has no automatic removal).
    """
    model = train_mod.init_model(args, args.model)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    phs, bottleneck_layers = train_mod.init_and_plug_phs(model, args, model_type=args.model)
    _warm_up_heads(model, phs, dls, args)
    if ph_paths is not None:
        for i, path in enumerate(ph_paths):
            checkpoint = torch.load(path, map_location=args.device, weights_only=True)
            try:
                phs[i].classifier.load_state_dict(checkpoint)
            except RuntimeError as exc:
                print(f"[WARNING] strict load of privacy head {i} failed ({exc}); "
                      "loading only shape-compatible parameters.")
                if hasattr(phs[i].classifier, "load_compatible_state_dict"):
                    phs[i].classifier.load_compatible_state_dict(checkpoint)
                else:
                    phs[i].classifier.load_state_dict(checkpoint, strict=False)
    return model, phs, bottleneck_layers


def online_batch_mi_training(model, phs, dls, criterion, args, cc, epochs, gamma_scalar,
                             tag, log_prefix):
    """Procedure ONLINEBATCHMITRAINING of the pseudocode.

    Per batch (inside cc.epoch_model_training): (1) update all privacy heads,
    (2) re-forward, (3) update backbone + task head on L_y + sum_i gamma_i I_z.
    """
    B = len(phs)
    gamma_vec = [gamma_scalar] * B
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.mom_sgd, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=cc.compute_exponential_gamma(args.lr, args.lr / 100.0, epochs))
    ph_optimizers = [torch.optim.SGD(ph.parameters(), lr=args.lr_ph[0],
                                     momentum=args.mom_sgd, weight_decay=args.wd)
                     for ph in phs]
    ph_gamma = cc.compute_exponential_gamma(args.lr_ph[0], args.lr_ph[0] / 100.0, epochs)
    ph_schedulers = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=ph_gamma)
                     for o in ph_optimizers]

    for e in range(epochs):
        train_log = cc.epoch_model_training(
            model=model, phs=phs, dls=dls, criterion=criterion,
            optimizer=optimizer, ph_optimizers=ph_optimizers, epoch=e, args=args,
            minimize_MI=True, train_model=True, train_PH=[True] * B,
            gamma=gamma_vec, printing=False,
        )
        val_log = cc.epoch_model_eval(model, phs, dls, criterion, args,
                                      minimize_MI=False, printing=False, split="val")
        last = B - 1
        print(f"[{tag}] epoch {e + 1}/{epochs}  train_acc={train_log['acc']:.2f}  "
              f"val_acc={val_log['acc']:.2f}  val_ph{last}_acc={val_log['ph'][last]['acc']:.2f}  "
              f"train_ph{last}_mi={train_log['ph'][last]['mi']:.4f}", flush=True)
        if wandb.run is not None:
            wandb.log({f"{log_prefix}/train": train_log, f"{log_prefix}/val": val_log,
                       f"{log_prefix}/epoch": e,
                       f"{log_prefix}/lr": optimizer.param_groups[0]["lr"]})
        scheduler.step()
        for s in ph_schedulers:
            s.step()
    return model, phs


def task_only_training(model, phs, dls, criterion, args, cc, epochs, log_prefix="task_pretrain"):
    """Phase 0: train f on the task only (no MI, privacy heads untouched).

    Returns the state_dict (CPU) of the best-val-accuracy epoch.
    """
    B = len(phs)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.mom_sgd, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=cc.compute_exponential_gamma(args.lr, args.lr / 100.0, max(epochs, 1)))

    best_acc, best_state = -1.0, _cpu_state(model)
    for e in range(epochs):
        train_log = cc.epoch_model_training(
            model=model, phs=phs, dls=dls, criterion=criterion,
            optimizer=optimizer, ph_optimizers=[None] * B, epoch=e, args=args,
            minimize_MI=False, train_model=True, train_PH=[False] * B,
            printing=False,
        )
        val_log = cc.epoch_model_eval(model, phs, dls, criterion, args,
                                      minimize_MI=False, printing=False, split="val")
        print(f"[{log_prefix}] epoch {e + 1}/{epochs}  train_acc={train_log['acc']:.2f}  "
              f"val_acc={val_log['acc']:.2f}", flush=True)
        if wandb.run is not None:
            wandb.log({f"{log_prefix}/train": train_log, f"{log_prefix}/val": val_log,
                       f"{log_prefix}/epoch": e,
                       f"{log_prefix}/lr": optimizer.param_groups[0]["lr"]})
        scheduler.step()
        if val_log["acc"] > best_acc:
            best_acc = val_log["acc"]
            best_state = _cpu_state(model)
    return best_state, best_acc


def apply_global_pruning(model, args, s, transformer_pruning):
    """Global structured pruning at sparsity s (identical to the original scripts):
    every Conv2d/Linear output channel is ranked by its L1 magnitude normalized by
    the channel size, and the lowest floor(s * N) channels are masked.
    """
    if transformer_pruning.enabled:
        total_channels, prune_count = transformer_pruning.prune(s)
        if total_channels == 0:
            raise RuntimeError(
                f"Transformer MLP pruning found no (fc1, fc2) pairs for model "
                f"'{args.model}': the sweep would silently run unpruned. "
                "(Known limitation: the HuggingFace 'vit' keeps its blocks at "
                "model.vit.encoder.layer, which transformer_mlp_pruning does not "
                "traverse; 'vit_b' and CNN backbones are fine.)")
        return total_channels, prune_count

    modules = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    global_channel_metrics = []
    for module in modules:
        weight = module.weight.detach()
        for channel_idx in range(weight.shape[0]):
            channel_weights = weight[channel_idx]
            magnitude = channel_weights.abs().sum().item() / channel_weights.numel()
            global_channel_metrics.append((magnitude, module, channel_idx))

    total_channels = len(global_channel_metrics)
    prune_count = int(np.floor(s * total_channels))
    if prune_count > 0:
        global_channel_metrics.sort(key=lambda x: x[0])
        channels_to_prune = defaultdict(list)
        for _, module, channel_idx in global_channel_metrics[:prune_count]:
            channels_to_prune[module].append(channel_idx)
        for module, channel_indices in channels_to_prune.items():
            mask = torch.ones_like(module.weight)
            for channel_idx in channel_indices:
                mask[channel_idx] = 0
            prune.custom_from_mask(module, name="weight", mask=mask)
    return total_channels, prune_count


def fold_pruning_masks(model, transformer_pruning):
    """Make pruning permanent: fold masks into the weights and drop the reparam,
    so the saved checkpoints load into vanilla (mask-free) models."""
    if transformer_pruning.enabled:
        transformer_pruning.clear_pruning()
        return
    for module in model.modules():
        if hasattr(module, "weight_orig"):
            prune.remove(module, "weight")


def attack_and_eval(model, phs, dls, criterion, args, cc, run_dir, e_ph, tag,
                    init_from_paths=None, splits=("train", "val", "test")):
    """Freeze f, retrain the privacy heads on D_train (the honest attack), then
    evaluate target and private accuracy on the requested splits."""
    phs, ph_paths = cc.train_privacy_heads(
        model=model, phs=phs, dls=dls, criterion=criterion, args=args,
        dir_name=run_dir, e_ph=e_ph, lr_ph=args.lr_ph[0],
        init_from_paths=init_from_paths, tag=tag,
    )
    logs = {}
    for split in splits:
        logs[split] = cc.epoch_model_eval(model, phs, dls, criterion, args,
                                          minimize_MI=False, printing=False, split=split)
    return phs, ph_paths, logs


def _row_from_logs(logs, B):
    last = B - 1
    row = {}
    for split, log in logs.items():
        row[f"target_acc_{split}"] = log["acc"]
        row[f"private_last_acc_{split}"] = log["ph"][last]["acc"]
        for i in range(B):
            row[f"ph{i}_acc_{split}"] = log["ph"][i]["acc"]
            row[f"ph{i}_mi_{split}"] = log["ph"][i]["mi"]
    return row


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------------
# Main — Algorithm 1
# ----------------------------------------------------------------------------

def main():
    args, dargs = _parse_driver_args()

    import pipeline as cc
    import train as train_mod
    from config import handle_all
    from dataloaders import build_dataloaders
    from pruning import eval_sparsity, eval_channel_sparsity
    from model_architectures.transformer_mlp_pruning import init_transformer_mlp_pruning

    if not torch.cuda.is_available():
        args.device = "cpu"

    # ---------------- data ----------------
    dls = {}
    dls["train"], dls["val"], dls["test"] = build_dataloaders(args)
    cc.print_sensitive_attribute_distribution(dls)
    cc.infer_private_num_classes(dls, args)
    cc.infer_target_num_classes(dls, args)
    print(f"Target classes: {args.target_num_classes} | Private classes: {args.private_num_classes}")

    criterion = nn.CrossEntropyLoss()

    # ---------------- run directory ----------------
    os.makedirs(args.save_path, exist_ok=True)
    cur_date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    run_name = f"{args.projectName}_{args.model}_{args.dataset}_seed{args.seed}_{cur_date}"
    run_dir = os.path.join(args.save_path, run_name)
    os.makedirs(run_dir, exist_ok=True)
    summary_path, stats_dir = cc.init_experiment_tracking(run_dir, args)
    args.output = stats_dir
    shutil.copyfile(os.path.abspath(__file__), os.path.join(run_dir, "driver_script.py"))
    print(f"Run directory: {run_dir}")

    if args.wandb_log:
        try:
            wandb.init(project=args.projectName, name=f"{run_name}_pseudocode",
                       config={"model": args.model, "dataset": args.dataset,
                               "seed": args.seed, "gamma_list": dargs.gamma_list,
                               "sparsity_list": dargs.sparsity_list, "t_ph": dargs.t_ph,
                               "e_task": dargs.e_task, "e_pre": dargs.e_pre,
                               "e_ft": dargs.e_ft, "e_ph": dargs.e_ph})
        except Exception as exc:  # keep the run alive without wandb
            print(f"[WARNING] wandb.init failed ({exc}); continuing without wandb.")

    # ---------------- model skeleton / per-head hyperparameter lists ----------------
    model, phs, bottleneck_layers = fresh_model_and_phs(args, cc, train_mod, dls)
    B = len(phs)
    handle_all(args, B)
    args.nb_epochs_ph = _as_list(args.nb_epochs_ph, int, B)
    args.nb_epochs_ph = args.nb_epochs_ph + [args.nb_epochs_ph[0]] * (B - len(args.nb_epochs_ph))
    args.lr_ph = _as_list(args.lr_ph, float, B)

    e_task = dargs.e_task if dargs.e_task is not None else args.nb_epochs
    e_pre = dargs.e_pre if dargs.e_pre is not None else args.nb_epochs
    e_ft = dargs.e_ft
    e_ph = dargs.e_ph if dargs.e_ph is not None else args.nb_epochs_ph[0]
    T_PH = dargs.t_ph
    last = B - 1

    pretrained_path = os.path.join(run_dir, "pretrained_model.pth")
    mi_model_path = os.path.join(run_dir, "mi_model.pth")
    pruned_model_path = os.path.join(run_dir, "pruned_model.pth")

    # =========================================================================
    # Phase 0 — task-only pretraining (the model "before the MI phase")
    # =========================================================================
    if dargs.reload_mi_model is not None and dargs.reload_task_model is None:
        task_state = None  # not needed: Phases 0-1 are skipped entirely
        cc.append_summary_line(summary_path, "Phase 0 skipped (reload_mi_model given).")
    elif dargs.reload_task_model is not None:
        task_state = torch.load(dargs.reload_task_model, map_location="cpu", weights_only=True)
        model.load_state_dict(task_state)
        torch.save(task_state, pretrained_path)
        cc.append_summary_line(summary_path,
                               f"Phase 0 skipped; task model reloaded from {dargs.reload_task_model}")
        print(f"[Phase 0] reloaded task model from {dargs.reload_task_model}")
    else:
        print(f"[Phase 0] task-only pretraining for {e_task} epochs")
        task_state, task_best_acc = task_only_training(model, phs, dls, criterion,
                                                       args, cc, e_task)
        torch.save(task_state, pretrained_path)
        cc.append_summary_line(summary_path,
                               f"Phase 0 done: best val acc {task_best_acc:.4f} -> {pretrained_path}")
        print(f"[Phase 0] saved {pretrained_path} (best val acc {task_best_acc:.2f})")
    del model, phs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # Phase 1 — GammaSelection
    # =========================================================================
    gamma_rows = []
    if dargs.reload_mi_model is not None:
        gamma_star = dargs.gamma_star
        mi_state = torch.load(dargs.reload_mi_model, map_location="cpu", weights_only=True)
        torch.save(mi_state, mi_model_path)
        # Reproduce the tail of GammaSelection for the reloaded backbone: freeze f
        # and retrain the privacy heads so the sweep can warm-start from them.
        model, phs, _ = fresh_model_and_phs(args, cc, train_mod, dls, state_dict=mi_state)
        phs, star_ph_paths, logs = attack_and_eval(model, phs, dls, criterion, args, cc,
                                                   run_dir, e_ph, tag="gamma_star_reload",
                                                   splits=("val",))
        a_star = logs["val"]["ph"][last]["acc"]
        cc.append_summary_line(summary_path,
                               f"Phase 1 skipped; MI model reloaded from {dargs.reload_mi_model} "
                               f"(gamma*={gamma_star:g}, last-head val acc {a_star:.2f})")
        del model, phs
    else:
        print(f"[Phase 1] GammaSelection over {dargs.gamma_list} ({e_pre} epochs each)")
        records = []
        for g in dargs.gamma_list:
            tag = f"gamma_{g:g}"
            model, phs, _ = fresh_model_and_phs(args, cc, train_mod, dls, state_dict=task_state)
            model, phs = online_batch_mi_training(model, phs, dls, criterion, args, cc,
                                                  e_pre, g, tag=tag, log_prefix=tag)
            mi_gamma_path = os.path.join(run_dir, f"mi_gamma_{g:g}.pth")
            torch.save(model.state_dict(), mi_gamma_path)
            # Freeze f^pre_gamma, retrain the privacy heads, measure a_gamma on D_val.
            phs, ph_paths, logs = attack_and_eval(model, phs, dls, criterion, args, cc,
                                                  run_dir, e_ph, tag=tag, splits=("val",))
            a_gamma = logs["val"]["ph"][last]["acc"]
            records.append({"gamma": g, "a_gamma": a_gamma, "mi_path": mi_gamma_path,
                            "ph_paths": ph_paths})
            gamma_rows.append({"gamma": g, "last_ph_val_acc": a_gamma,
                               "target_val_acc": logs["val"]["acc"],
                               "mi_checkpoint": mi_gamma_path, "selected": False})
            _write_csv(os.path.join(stats_dir, "gamma_selection.csv"), gamma_rows)
            print(f"[Phase 1] gamma={g:g}: last-head val acc {a_gamma:.2f}, "
                  f"target val acc {logs['val']['acc']:.2f}")
            if wandb.run is not None:
                wandb.log({"gamma_selection/gamma": g, "gamma_selection/a_gamma": a_gamma,
                           "gamma_selection/target_val_acc": logs["val"]["acc"]})
            del model, phs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        best = min(records, key=lambda r: r["a_gamma"])
        gamma_star = best["gamma"]
        star_ph_paths = best["ph_paths"]
        mi_state = torch.load(best["mi_path"], map_location="cpu", weights_only=True)
        torch.save(mi_state, mi_model_path)
        for row in gamma_rows:
            row["selected"] = (row["gamma"] == gamma_star)
        _write_csv(os.path.join(stats_dir, "gamma_selection.csv"), gamma_rows)
        cc.append_summary_line(summary_path,
                               f"Phase 1 done: gamma*={gamma_star:g} "
                               f"(last-head val acc {best['a_gamma']:.2f}) -> {mi_model_path}")
        print(f"[Phase 1] gamma* = {gamma_star:g} -> {mi_model_path}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # Phase 2 — sparsity sweep on f^pre_{gamma*}
    # =========================================================================
    print(f"[Phase 2] sparsity sweep over {dargs.sparsity_list} "
          f"(gamma*={gamma_star:g}, {e_ft} fine-tuning epochs each)")
    sweep_rows, sweep_records = [], []
    for s in dargs.sparsity_list:
        tag = f"s_{s:g}"
        model, phs, _ = fresh_model_and_phs(args, cc, train_mod, dls, state_dict=mi_state,
                                            ph_paths=star_ph_paths)
        transformer_pruning = init_transformer_mlp_pruning(model, args.model)
        total_channels, prune_count = apply_global_pruning(model, args, s, transformer_pruning)
        print(f"[Phase 2] s={s:g}: masked {prune_count}/{total_channels} channels")

        if e_ft > 0:
            model, phs = online_batch_mi_training(model, phs, dls, criterion, args, cc,
                                                  e_ft, gamma_star, tag=tag,
                                                  log_prefix=f"sweep/{tag}")
        fold_pruning_masks(model, transformer_pruning)

        # Honest attack on the frozen pruned backbone + full evaluation.
        phs, _, logs = attack_and_eval(model, phs, dls, criterion, args, cc,
                                       run_dir, e_ph, tag=tag,
                                       splits=("train", "val", "test"))

        global_s, _ = eval_sparsity(model, args)
        _, channel_summary = eval_channel_sparsity(
            model, args=args, sort_by="sparsity", descending=True, top_k=0,
            csv_path=os.path.join(stats_dir, f"channel_sparsity_{tag}.csv"),
            wandb_prefix=f"sparsity/{tag}")

        ckpt_path = os.path.join(run_dir, f"pruned_s_{s:g}.pth")
        torch.save(model.state_dict(), ckpt_path)

        priv_val_frac = _acc_to_frac(logs["val"]["ph"][last]["acc"])
        row = {"s_nominal": s,
               "channels_pruned": prune_count,
               "channels_total": total_channels,
               "global_weight_sparsity": global_s,
               "channel_sparsity_measured": channel_summary["relative_sparsity"],
               **_row_from_logs(logs, B),
               "accepted_priv_below_t_ph": priv_val_frac < T_PH,
               "selected": False,
               "checkpoint": ckpt_path}
        sweep_rows.append(row)
        sweep_records.append({"s": s, "ckpt": ckpt_path,
                              "target_val": logs["val"]["acc"],
                              "priv_val_frac": priv_val_frac})
        _write_csv(os.path.join(stats_dir, "results_sparsity_sweep.csv"), sweep_rows)
        print(f"[Phase 2] s={s:g}  target val acc={logs['val']['acc']:.2f}  "
              f"last-head val acc={logs['val']['ph'][last]['acc']:.2f}  "
              f"accepted={priv_val_frac < T_PH}")
        if wandb.run is not None:
            wandb.log({"sweep/s": s, "sweep/target_val_acc": logs["val"]["acc"],
                       "sweep/priv_last_val_acc": logs["val"]["ph"][last]["acc"],
                       "sweep/target_test_acc": logs["test"]["acc"],
                       "sweep/priv_last_test_acc": logs["test"]["ph"][last]["acc"],
                       "sweep/global_weight_sparsity": global_s})
        del model, phs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =========================================================================
    # Phase 3 — selection (C_priv filter, then argmax target val accuracy)
    # =========================================================================
    c_priv = [r for r in sweep_records if r["priv_val_frac"] < T_PH]
    if c_priv:
        chosen = max(c_priv, key=lambda r: r["target_val"])
        reason = f"highest target val acc among priv < {T_PH:g}"
    else:
        chosen = min(sweep_records, key=lambda r: r["priv_val_frac"])
        reason = f"no candidate below T_PH={T_PH:g}; lowest last-head val acc"
    shutil.copyfile(chosen["ckpt"], pruned_model_path)
    for row in sweep_rows:
        row["selected"] = (row["s_nominal"] == chosen["s"])
    _write_csv(os.path.join(stats_dir, "results_sparsity_sweep.csv"), sweep_rows)

    # Artifacts of skipped phases must not be advertised as deliverables.
    pretrained_out = pretrained_path if os.path.exists(pretrained_path) else None
    gamma_csv = os.path.join(stats_dir, "gamma_selection.csv")
    gamma_csv_out = gamma_csv if os.path.exists(gamma_csv) else None

    cc.append_summary_line(summary_path, f"Phase 3: selected s={chosen['s']:g} ({reason})")
    cc.append_summary_line(summary_path,
                           f"pretrained_model: {pretrained_out or 'not produced (Phase 0 skipped)'}")
    cc.append_summary_line(summary_path, f"mi_model (gamma*={gamma_star:g}): {mi_model_path}")
    cc.append_summary_line(summary_path, f"pruned_model (s={chosen['s']:g}): {pruned_model_path}")

    manifest = {"gamma_star": gamma_star, "s_selected": chosen["s"],
                "selection_reason": reason,
                "pretrained_model": pretrained_out,
                "mi_model": mi_model_path,
                "pruned_model": pruned_model_path,
                "gamma_selection_csv": gamma_csv_out,
                "results_csv": os.path.join(stats_dir, "results_sparsity_sweep.csv")}
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n================ CutClean pseudocode run complete ================")
    print(f"gamma*         : {gamma_star:g}")
    print(f"selected s     : {chosen['s']:g}  ({reason})")
    print(f"pretrained     : {pretrained_out or 'not produced (Phase 0 skipped)'}")
    print(f"MI (unpruned)  : {mi_model_path}")
    print(f"pruned (final) : {pruned_model_path}")
    print(f"results        : {manifest['results_csv']}")
    if wandb.run is not None:
        wandb.log({"final/gamma_star": gamma_star, "final/s_selected": chosen["s"]})
        wandb.finish()
    return manifest


if __name__ == "__main__":
    main()
