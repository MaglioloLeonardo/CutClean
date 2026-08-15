import csv
import time
from collections import Counter, defaultdict

from dataloaders import build_dataloaders
from irene.utilities import AverageMeter, accuracy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from tqdm import tqdm
import wandb
import os
from config import parse_args
import re
import sys
import shutil
from datetime import datetime
from pruning import eval_channel_sparsity, eval_sparsity, prune_model
from model_architectures.transformer_mlp_pruning import init_transformer_mlp_pruning
from torch.amp import autocast
from contextlib import nullcontext

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from train import init_model, init_and_plug_phs, compute_MI
from config import handle_all

PROJECT = "CutCleanResnetAll1PHCorrected100"


def check_labels(output, target, name):
    """
    Check that targets are compatible with a CrossEntropyLoss output:
    - integer dtype
    - values in [0, num_classes-1]
    """
    num_classes = output.size(1)

    if target.dtype not in (torch.int32, torch.int64, torch.long):
        raise ValueError(
            f"{name}: dtype {target.dtype} non valido per CrossEntropyLoss "
            "(usa long/int64)."
        )

    t_min = int(target.min().item())
    t_max = int(target.max().item())

    if t_min < 0 or t_max >= num_classes:
        raise ValueError(
            f"{name}: valori fuori range per CrossEntropyLoss "
            f"(min={t_min}, max={t_max}, num_classes={num_classes})."
        )


def freeze_model_parameters(model: torch.nn.Module) -> None:
    """Freeze all parameters of the main model f (encoder + task head gc)."""
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_model_parameters(model: torch.nn.Module) -> None:
    """Unfreeze all parameters of the main model f (encoder + task head gc)."""
    for p in model.parameters():
        p.requires_grad = True


def init_experiment_tracking(run_dir, args):
    stats_dir = os.path.join(run_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    summary_path = os.path.join(stats_dir, "run_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write("=== Configurazione esperimento ===\n")
        summary_file.write(f"Timestamp: {datetime.now().isoformat()}\n")
        summary_file.write("Argomenti da console:\n")
        summary_file.write(" ".join(sys.argv) + "\n\n")
        summary_file.write("Parametri parsati:\n")
        for key in sorted(vars(args).keys()):
            summary_file.write(f"- {key}: {getattr(args, key)}\n")
        summary_file.write("\n")
    experiment_script_path = os.path.join(run_dir, "experiment_script.py")
    shutil.copyfile(os.path.abspath(__file__), experiment_script_path)
    return summary_path, stats_dir


def append_summary_line(summary_path, text):
    with open(summary_path, "a", encoding="utf-8") as summary_file:
        summary_file.write(text + "\n")


def log_block_summary(summary_path, block_idx, best_sparsity, best_val_acc):
    append_summary_line(
        summary_path,
        f"Blocco {block_idx}: sparsity ottimale {best_sparsity:.6f}, val_acc {best_val_acc:.4f}",
    )


def log_final_summary(summary_path, best_model_path, block_results, eval_logs):
    append_summary_line(summary_path, "\n=== Modello finale ricaricato ===")
    append_summary_line(summary_path, f"Checkpoint: {best_model_path}")
    sparsity_signature = ", ".join(
        [
            f"block{idx}={info['sparsity']:.6f}"
            for idx, info in sorted(block_results.items())
        ]
    )
    append_summary_line(summary_path, f"Firma sparsity: {sparsity_signature}")
    append_summary_line(summary_path, "\n=== Metriche finali (modello ricaricato) ===")
    for split_label, log in eval_logs.items():
        append_summary_line(
            summary_path,
            f"{split_label}: acc={log['acc']:.4f}, loss={log['loss']:.4f}",
        )


def log_full_config_to_wandb(args):
    if wandb.run is None:
        return
    try:
        wandb.config.update(vars(args), allow_val_change=True)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Impossibile sincronizzare la config completa su wandb: {exc}")


def save_split_stats(stats_dir, split_label, block_tracking, eval_log):
    os.makedirs(stats_dir, exist_ok=True)
    split_prefix = f"final/{split_label.lower().replace(' ', '_')}"
    ph_indices = sorted(eval_log.get("ph", {}).keys())
    fieldnames = ["block", "sparsity_ottimale", "val_acc"]
    for ph_idx in ph_indices:
        fieldnames.append(f"ph{ph_idx}_MI_after_ph_finetune")
        fieldnames.append(f"ph{ph_idx}_Acc_after_ph_finetune")

    fieldnames.extend(
        [
            f"{split_prefix}/model/accuracy",
            f"{split_prefix}/model/loss",
        ]
    )
    for ph_idx in ph_indices:
        fieldnames.extend(
            [
                f"{split_prefix}/privacy_head_{ph_idx}/accuracy",
                f"{split_prefix}/privacy_head_{ph_idx}/loss",
                f"{split_prefix}/privacy_head_{ph_idx}/mutual_info",
            ]
        )

    rows = []
    for block_idx, info in sorted(block_tracking.items()):
        row = {
            "block": block_idx,
            "sparsity_ottimale": info.get("sparsity", 0),
            "val_acc": info.get("val_acc", 0),
        }
        row.update(
            {
                f"{split_prefix}/model/accuracy": eval_log.get("acc"),
                f"{split_prefix}/model/loss": eval_log.get("loss"),
            }
        )
        for ph_idx in ph_indices:
            ph_metrics = eval_log["ph"].get(ph_idx, {})
            row[f"ph{ph_idx}_MI_after_ph_finetune"] = ph_metrics.get("mi")
            row[f"ph{ph_idx}_Acc_after_ph_finetune"] = ph_metrics.get("acc")
            row[f"{split_prefix}/privacy_head_{ph_idx}/accuracy"] = ph_metrics.get("acc")
            row[f"{split_prefix}/privacy_head_{ph_idx}/loss"] = ph_metrics.get("loss")
            row[f"{split_prefix}/privacy_head_{ph_idx}/mutual_info"] = ph_metrics.get("mi")
        rows.append(row)

    stats_file = os.path.join(
        stats_dir, f"{split_label.lower().replace(' ', '_')}_stats.csv"
    )
    with open(stats_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_all_stats(stats_dir, block_tracking, eval_logs):
    for split_label, log in eval_logs.items():
        save_split_stats(stats_dir, split_label, block_tracking, log)


def build_block_sparsity_tag(block_idx, current_sparsity, block_tracking):
    """Create a descriptive tag encoding the sparsity history up to a block."""

    history = [
        block_tracking[i]["sparsity"]
        for i in range(block_idx)
        if i in block_tracking
    ]
    history.append(current_sparsity)
    history_str = "-".join(f"{value:.6f}" for value in history)
    return f"block_{block_idx}_sparsity_{current_sparsity:.6f}_sparsities{history_str}"


def create_run_config(args, experiment_type, block=None, sparsity=None):
    run_name_parts = [
        args.model,
        args.dataset,
        f"seed{args.seed}",
        experiment_type
    ]
    if block is not None:
        run_name_parts.append(f"block{block}")
    if sparsity is not None:
        run_name_parts.append(f"spars{sparsity}")
    run_name = "_".join(run_name_parts)
    if experiment_type == "raw_model":
        group = f"baselines_{args.model}_{args.dataset}"
    elif experiment_type.startswith("block"):
        group = f"pruning_{args.model}_{args.dataset}_seed{args.seed}"
    elif experiment_type == "finetuning":
        group = f"finetuning_{args.model}_{args.dataset}_seed{args.seed}"
    return run_name, group


def log_comprehensive_metrics(epoch, model_metrics, ph_metrics, sparsity_info=None,
                              training_phase="", prefix=""):
    log_dict = {}
    log_dict.update({
        f"{prefix}model/accuracy": model_metrics["acc"],
        f"{prefix}model/loss": model_metrics["loss"],
    })
    for i, ph_metric in ph_metrics.items():
        log_dict.update({
            f"{prefix}privacy_head_{i}/accuracy": ph_metric["acc"],
            f"{prefix}privacy_head_{i}/loss": ph_metric["loss"],
            f"{prefix}privacy_head_{i}/mutual_info": ph_metric["mi"],
        })
    max_ph_acc = max([ph["acc"] for ph in ph_metrics.values()])
    avg_mi = np.mean([ph["mi"] for ph in ph_metrics.values()])
    log_dict.update({
        f"{prefix}privacy/max_ph_accuracy": max_ph_acc,
        f"{prefix}privacy/avg_mutual_info": avg_mi,
        f"{prefix}privacy/privacy_score": 100 - max_ph_acc,
    })
    if sparsity_info:
        log_dict.update({
            f"{prefix}sparsity/current_block": sparsity_info.get("block", -1),
            f"{prefix}sparsity/current_sparsity": sparsity_info.get("sparsity", 0),
            f"{prefix}sparsity/total_params_pruned": sparsity_info.get("total_pruned", 0),
        })
    log_dict.update({
        f"{prefix}training/epoch": epoch,
        f"{prefix}training/phase": training_phase,
    })
    wandb.log(log_dict)


def log_split_metrics(split_name, log_data):
    if wandb.run is None:
        return
    payload = {
        f"{split_name}/model/accuracy": log_data.get("acc"),
        f"{split_name}/model/loss": log_data.get("loss"),
    }
    for ph_idx, ph_metric in log_data.get("ph", {}).items():
        payload.update(
            {
                f"{split_name}/privacy_head_{ph_idx}/accuracy": ph_metric.get("acc"),
                f"{split_name}/privacy_head_{ph_idx}/loss": ph_metric.get("loss"),
                f"{split_name}/privacy_head_{ph_idx}/mutual_info": ph_metric.get("mi"),
            }
        )
    wandb.log(payload)


def log_final_wandb_summary(eval_logs, last_ph_idx, global_sparsity, channel_relative_sparsity):
    """
    Log a compact summary after the final privacy-head retraining.

    Tracks, per split, the task accuracy and the accuracy of the last privacy head,
    plus the achieved sparsity levels.
    """
    if wandb.run is None:
        return

    payload = {
        "final_summary/sparsity/global_fraction": global_sparsity,
        "final_summary/sparsity/channel_relative_fraction": channel_relative_sparsity,
    }

    for split_label, log_data in eval_logs.items():
        split_key = split_label.lower().replace(" ", "_")
        split_prefix = f"final_summary/{split_key}"
        payload[f"{split_prefix}/target_accuracy"] = log_data.get("acc")
        if last_ph_idx is not None:
            payload[f"{split_prefix}/privacy_head_{last_ph_idx}_accuracy"] = (
                log_data.get("ph", {}).get(last_ph_idx, {}).get("acc")
            )

    wandb.log(payload)
    for key, value in payload.items():
        wandb.run.summary[key] = value


def _collect_target_bias_pairs(dataset):
    """Collect target and sensitive-attribute labels without loading images."""

    if hasattr(dataset, "attr_df") and hasattr(dataset, "target") and hasattr(dataset, "bias_attr"):
        targets = dataset.attr_df[dataset.target].tolist()
        biases = dataset.attr_df[dataset.bias_attr].tolist()
    elif hasattr(dataset, "targets") and hasattr(dataset, "bias_targets"):
        targets = dataset.targets.detach().cpu().numpy().tolist()
        biases = dataset.bias_targets.detach().cpu().numpy().tolist()
    elif hasattr(dataset, "paths"):
        targets, biases = [], []
        for path in dataset.paths:
            name = os.path.basename(path)
            parts = name.split("_")
            try:
                targets.append(int(parts[-2].replace("lbl", "")))
                biases.append(int(parts[-1].replace("bias", "").replace(".png", "")))
            except (IndexError, ValueError):  # pragma: no cover - defensive
                continue
    elif hasattr(dataset, "data") and isinstance(dataset.data, (list, tuple)):
        targets, biases = [], []
        for path in dataset.data:
            # CIFAR10C stores target and bias in the filename: ..._<target>_<bias>.jpg
            try:
                parts = os.path.basename(path).split("_")
                targets.append(int(parts[-2]))
                biases.append(int(parts[-1].split(".")[0]))
            except (IndexError, ValueError):  # pragma: no cover - defensive
                continue
    else:
        targets, biases = [], []
        for sample in dataset:
            if len(sample) < 2:
                continue

            target = sample[1]
            bias = sample[2] if len(sample) > 2 else None

            # Some datasets return (img, (label, bias), idx)
            if isinstance(target, (list, tuple)) and len(target) >= 2:
                bias = target[1]
                target = target[0]

            if bias is None:
                continue

            targets.append(int(target))
            biases.append(int(bias))
    return targets, biases


def print_sensitive_attribute_distribution(dls):
    """Print the sensitive-attribute distribution per class for each split."""

    for split_label in ["train", "val", "test"]:
        if split_label not in dls:
            continue
        dataset = dls[split_label].dataset
        targets, biases = _collect_target_bias_pairs(dataset)
        print(f"\n=== Distribuzione attributo sensibile: {split_label} ===")
        if not targets:
            print("Nessun dato disponibile per calcolare la distribuzione.")
            continue

        per_class_counts = defaultdict(Counter)
        for target, bias in zip(targets, biases):
            per_class_counts[int(target)][int(bias)] += 1

        all_bias_values = sorted({b for counts in per_class_counts.values() for b in counts})
        for target_class in sorted(per_class_counts.keys()):
            counts = per_class_counts[target_class]
            total = sum(counts.values())
            bias_parts = [f"bias {bias_val}: {counts[bias_val]}" for bias_val in all_bias_values]
            print(
                f"Classe {target_class} (totale {total}): "
                + ", ".join(bias_parts)
            )


def infer_private_num_classes(dls, args):
    """
    Set args.private_num_classes from the private-label values found in the
    training set (works for both CelebA and corrupted CIFAR).
    """
    train_dataset = dls["train"].dataset
    _, biases = _collect_target_bias_pairs(train_dataset)

    if not biases:
        # ultra-defensive fallback: scan a handful of samples one by one
        private_values = set()
        for i in range(min(len(train_dataset), 1024)):
            _, _, priv = train_dataset[i]
            private_values.add(int(priv))
        biases = list(private_values)

    if biases:
        args.private_num_classes = int(max(biases)) + 1
    else:
        # binary default when nothing at all could be found
        if not hasattr(args, "private_num_classes"):
            args.private_num_classes = 2


def infer_target_num_classes(dls, args):
    """Infer the number of classes of the main task by scanning the labels."""

    train_dataset = dls["train"].dataset
    targets, _ = _collect_target_bias_pairs(train_dataset)

    if not targets:
        # fallback: sample a few examples directly from the dataset
        sampled_targets = set()
        for i in range(min(len(train_dataset), 1024)):
            _, tgt, _ = train_dataset[i]
            sampled_targets.add(int(tgt))
        targets = list(sampled_targets)

    if targets:
        args.target_num_classes = int(max(targets)) + 1
    elif not hasattr(args, "target_num_classes"):
        args.target_num_classes = 2


def init_wandb_run(args, experiment_type, block=None, sparsity=None):
    if wandb.run is not None:
        wandb.finish()
    run_name, group = create_run_config(args, experiment_type, block, sparsity)
    gamma_tag = f"gamma_{'-'.join([str(g) for g in args.gamma])}"
    tags = [
        args.model,
        args.dataset,
        experiment_type,
        gamma_tag,
    ]
    if block is not None:
        tags.append(f"block_{block}")
    config = {
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "gamma": args.gamma,
        "used_phs": args.used_phs,
        "lr": args.lr,
        "lr_ph": args.lr_ph,
        "nb_epochs": getattr(args, 'nb_epochs', None),
        "nb_epochs_ph": getattr(args, 'nb_epochs_ph', None),
        "experiment_type": experiment_type,
        "current_block": block,
        "current_sparsity": sparsity,
        "num_blocks": len(bottleneck_layers) if 'bottleneck_layers' in locals() else None,
    }
    wandb.init(
        project=args.projectName,
        name=run_name,
        group=group,
        tags=tags,
        config=config,
        reinit=True,
    )
    log_full_config_to_wandb(args)


def log_phase_summary(phase_name, final_metrics, sparsities=None):
    summary_dict = {
        f"summary/{phase_name}_final_accuracy": final_metrics["acc"],
        f"summary/{phase_name}_privacy_preservation": 100 - max([ph["acc"] for ph in final_metrics["ph"].values()]),
    }
    if sparsities:
        summary_dict[f"summary/{phase_name}_total_sparsity"] = sum(sparsities.values()) / len(sparsities)
    wandb.log(summary_dict)
    for key, value in summary_dict.items():
        wandb.run.summary[key] = value


def compute_joint_loss(loss_task, mi_phs, args, loss_type="irene", gamma=None):
    if gamma is None:
        gamma = args.gamma
    if loss_type == "irene":
        loss = args.alpha * loss_task
        for i in range(len(gamma)):
            loss += gamma[i] * mi_phs[i]
    return loss


def compute_exponential_gamma(lr_start, lr_end, epochs):
    if epochs <= 0 or lr_start <= 0 or lr_end <= 0:
        return 1.0
    return (lr_end / lr_start) ** (1.0 / epochs)


def _ensure_tensor_on_device(value, device):
    """Convert lists or tuples of labels to tensors and move them to the device."""

    if torch.is_tensor(value):
        return value.to(device)

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return torch.empty(0, device=device)

        if all(torch.is_tensor(v) for v in value):
            try:
                value = torch.stack(value)
            except RuntimeError:
                value = torch.cat([v.unsqueeze(0) for v in value])
        else:
            value = torch.as_tensor(value)

    return torch.as_tensor(value).to(device)


def _prepare_targets_for_loss(target: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Ensure targets line up with the batch dimension expected by CrossEntropyLoss."""

    if target.dim() > 1:
        # Targets sometimes arrive as (num_labels, batch) instead of (batch, num_labels).
        if target.shape[0] != batch_size and target.shape[-1] == batch_size:
            target = target.transpose(0, -1)

        # Convert one-hot or probability targets to class indices.
        if target.shape[0] == batch_size and target.shape[1] > 1:
            target = target.argmax(dim=1)
        elif target.shape[0] == batch_size:
            target = target.view(batch_size)

    if target.shape[0] != batch_size:
        raise ValueError(
            f"Target batch dimension mismatch: expected {batch_size}, got {tuple(target.shape)}"
        )

    return target


def epoch_model_training(
        model,
        phs,
        dls,
        criterion,
        optimizer,
        ph_optimizers,
        epoch,
        args,
        minimize_MI,
        train_model,
        train_PH,
        mode="MI",
        printing=True,
        gamma=None,
):
    model.train()
    for ph in phs:
        ph.train()
    if gamma is None:
        gamma = args.gamma
    tk = tqdm(dls["train"], total=int(len(dls["train"])), leave=printing)
    if printing:
        tk.set_description(f"Epoch {epoch + 1:>3}/{args.nb_epochs}")
    tot_acc = AverageMeter("acc")
    tot_loss = AverageMeter("loss")
    tot_acc_phs = {i: AverageMeter(f"acc_ph_{i}") for i in range(len(phs))}
    tot_loss_phs = {i: AverageMeter(f"loss_phs_{i}") for i in range(len(phs))}
    tot_mi = {i: AverageMeter(f"mi_ph_{i}") for i in range(len(phs))}
    for batch, (data, target, private_label) in enumerate(tk):
        data = data.to(args.device)
        target = _prepare_targets_for_loss(
            _ensure_tensor_on_device(target, args.device), data.size(0)
        )
        private_label = _prepare_targets_for_loss(
            _ensure_tensor_on_device(private_label, args.device), data.size(0)
        )

        # 1) Train the privacy heads on the current batch.
        if args.model == "vit":
            forward_model = model(data)
            _ = forward_model.logits
        else:
            _ = model(data)

        loss_phs = {i: 0 for i in range(len(phs))}
        for i, ph in enumerate(phs):
            ph_output = ph()
            check_labels(ph_output, private_label, f"private_label PH{i} (train step 1)")
            loss_phs[i] = criterion(ph_output, private_label)
        for i, ph_optimizer in enumerate(ph_optimizers):
            if train_PH[i]:
                ph_optimizer.zero_grad()
                loss_phs[i].backward()
                ph_optimizer.step()

        # 2) Recompute outputs to measure MI after updating the privacy heads.
        if args.model == "vit":
            forward_model = model(data)
            output = forward_model.logits
        else:
            output = model(data)

        # 3) Train the backbone with MI minimization from step 2 on the same batch.
        check_labels(output, target, "main target (train)")
        loss_task = criterion(output, target)
        acc = accuracy(output, target)[0].item()
        tot_acc.update(acc, data.size(0))

        loss_phs_eval = {i: 0 for i in range(len(phs))}
        mi_phs = {i: 0 for i in range(len(phs))}
        for i, ph in enumerate(phs):
            ph_output = ph()
            check_labels(ph_output, private_label, f"private_label PH{i} (train step 2)")
            loss_phs_eval[i] = criterion(ph_output, private_label)
            tot_loss_phs[i].update(loss_phs_eval[i].item(), data.size(0))
            acc_ph_i = accuracy(ph_output, private_label)[0].item()
            tot_acc_phs[i].update(acc_ph_i, data.size(0))
            nb_priv = getattr(args, "private_num_classes", 2)
            mi_phs[i] = compute_MI(private_label, ph, nb_classes=nb_priv, args=args)
            tot_mi[i].update(mi_phs[i].item(), data.size(0))

        if minimize_MI:
            loss = compute_joint_loss(
                loss_task, mi_phs, args, loss_type="irene", gamma=gamma
            )
        else:
            loss = loss_task

        tot_loss.update(loss.item(), data.size(0))

        if train_model:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if printing:
            tk.set_postfix(
                {
                    "loss": round(tot_loss.avg, 4),
                    "acc": round(tot_acc.avg, 4),
                    "phs_a": {i: round(tot_acc_phs[i].avg, 3) for i in range(len(phs))},
                    "phs_l": {
                        i: round(tot_loss_phs[i].avg, 3) for i in range(len(phs))
                    },
                    "phs_mi": {i: round(tot_mi[i].avg, 4) for i in range(len(phs))},
                }
            )
    train_log = {
        "loss": tot_loss.avg,
        "acc": tot_acc.avg,
        "ph": {
            i: {
                "acc": tot_acc_phs[i].avg,
                "loss": tot_loss_phs[i].avg,
                "mi": tot_mi[i].avg,
            }
            for i in range(len(phs))
        },
    }
    return train_log


import re


def epoch_ph_training(
        model,
        phs,
        dls,
        criterion,
        ph_optimizers_epoch,
        args,
        to_train=0,
        epoch_ph_begin=0,
        epoch_ph=0,
):
    model.eval()
    if type(to_train) == int:
        to_train = [to_train]
    elif to_train == "all":
        to_train = list(range(len(phs)))
    for i, ph in enumerate(phs):
        if i in to_train:
            ph.train()
        else:
            ph.eval()
    tk = tqdm(dls["train"], total=int(len(dls["train"])), leave=True)
    tk.set_description(
        f"Epoch {epoch_ph :>3}/{epoch_ph_begin + args.nb_epochs_ph[to_train[0]]}"
    )
    tot_acc_phs = {i: AverageMeter(f"acc_ph_{i}") for i in range(len(phs))}
    tot_loss_phs = {i: AverageMeter(f"loss_phs_{i}") for i in range(len(phs))}
    tot_mi = {i: AverageMeter(f"mi_ph_{i}") for i in range(len(phs))}
    for _, (data, _, private_label) in enumerate(tk):
        data = data.to(args.device)
        private_label = _prepare_targets_for_loss(
            _ensure_tensor_on_device(private_label, args.device), data.size(0)
        )
        if args.model == "vit":
            forward_model = model(data)
            _ = forward_model.logits
        else:
            _ = model(data)
        loss_phs = {i: 0 for i in range(len(phs))}
        mi_phs = {i: 0 for i in range(len(phs))}
        for i, ph in enumerate(phs):
            if i in to_train:
                ph_output = ph()
                check_labels(ph_output, private_label, f"private_label PH{i} (epoch_ph_training)")
                loss_phs[i] = criterion(ph_output, private_label)
                tot_loss_phs[i].update(loss_phs[i].item(), data.size(0))
                acc_ph_i = accuracy(ph_output, private_label)[0].item()
                tot_acc_phs[i].update(acc_ph_i, data.size(0))
                nb_priv = getattr(args, "private_num_classes", 2)
                mi_phs[i] = compute_MI(private_label, ph, nb_classes=nb_priv, args=args)
                tot_mi[i].update(mi_phs[i].item(), data.size(0))
        for i, ph_optimizer in enumerate(ph_optimizers_epoch):
            if i in to_train:
                ph_optimizer.zero_grad()
                loss_phs[i].backward()
                ph_optimizer.step()
        tk.set_postfix(
            {
                "phs_a": {i: round(tot_acc_phs[i].avg, 2) for i in range(len(phs))},
                "phs_l": {i: round(tot_loss_phs[i].avg, 2) for i in range(len(phs))},
                "phs_mi": {i: round(tot_mi[i].avg, 4) for i in range(len(phs))},
            }
        )
    ph_train_log = {
        "ph": {
            i: {
                "acc": tot_acc_phs[i].avg,
                "loss": tot_loss_phs[i].avg,
                "mi": tot_mi[i].avg,
            }
            for i in to_train
        },
    }
    return ph_train_log


def epoch_eval_ph(
        model,
        phs,
        dls,
        criterion,
        args,
        to_train=0,
        split="val",
):
    amp_ctx = autocast("cuda") if torch.cuda.is_available() else nullcontext()
    with torch.no_grad(), amp_ctx:
        model.eval()
        if type(to_train) == int:
            to_train = [to_train]
        elif to_train == "all":
            to_train = list(range(len(phs)))
        for i, ph in enumerate(phs):
            ph.eval()
        tk = tqdm(dls[split], total=int(len(dls[split])), leave=True)
        tot_acc_phs = {i: AverageMeter(f"acc_ph_{i}") for i in range(len(phs))}
        tot_loss_phs = {i: AverageMeter(f"loss_phs_{i}") for i in range(len(phs))}
        tot_mi = {i: AverageMeter(f"mi_ph_{i}") for i in range(len(phs))}
        for _, (data, _, private_label) in enumerate(tk):
            data = data.to(args.device)
            private_label = _prepare_targets_for_loss(
                _ensure_tensor_on_device(private_label, args.device), data.size(0)
            )
            if args.model == "vit":
                forward_model = model(data)
                _ = forward_model.logits
            else:
                _ = model(data)
            loss_phs = {i: 0 for i in range(len(phs))}
            mi_phs = {i: 0 for i in range(len(phs))}
            for i, ph in enumerate(phs):
                if i in to_train:
                    ph_output = ph()
                    check_labels(ph_output, private_label, f"private_label PH{i} (epoch_eval_ph, split={split})")
                    loss_phs[i] = criterion(ph_output, private_label)
                    tot_loss_phs[i].update(loss_phs[i].item(), data.size(0))
                    acc_ph_i = accuracy(ph_output, private_label)[0].item()
                    tot_acc_phs[i].update(acc_ph_i, data.size(0))
                    nb_priv = getattr(args, "private_num_classes", 2)
                    mi_phs[i] = compute_MI(private_label, ph, nb_classes=nb_priv, args=args)
                    tot_mi[i].update(mi_phs[i].item(), data.size(0))
            tk.set_postfix(
                {
                    "phs_a": {i: round(tot_acc_phs[i].avg, 2) for i in range(len(phs))},
                    "phs_l": {
                        i: round(tot_loss_phs[i].avg, 2) for i in range(len(phs))
                    },
                    "phs_mi": {i: round(tot_mi[i].avg, 4) for i in range(len(phs))},
                }
            )
        ph_val_log = {
            "ph": {
                i: {
                    "acc": tot_acc_phs[i].avg,
                    "loss": tot_loss_phs[i].avg,
                    "mi": tot_mi[i].avg,
                }
                for i in to_train
            },
        }
        return ph_val_log


def epoch_model_eval(model, phs, dls, criterion, args, minimize_MI=True, printing=True, split="val"):
    model.eval()
    for ph in phs:
        ph.eval()
    amp_ctx = autocast("cuda") if torch.cuda.is_available() else nullcontext()
    with torch.no_grad(), amp_ctx:
        tk = tqdm(dls[split], total=int(len(dls[split])), leave=printing)
        if printing:
            tk.set_description(f"------- Eval")
        tot_acc = AverageMeter("acc")
        tot_loss = AverageMeter("loss")
        tot_acc_phs = {i: AverageMeter(f"acc_ph_{i}") for i in range(len(phs))}
        tot_loss_phs = {i: AverageMeter(f"loss_phs_{i}") for i in range(len(phs))}
        tot_mi = {i: AverageMeter(f"mi_ph_{i}") for i in range(len(phs))}
        mi_ph = torch.zeros(len(phs), device=args.device)
        for batch, (data, target, private_label) in enumerate(tk):
            data = data.to(args.device)
            target = _prepare_targets_for_loss(
                _ensure_tensor_on_device(target, args.device), data.size(0)
            )
            private_label = _prepare_targets_for_loss(
                _ensure_tensor_on_device(private_label, args.device), data.size(0)
            )
            if args.model == "vit":
                forward_model = model(data)
                output = forward_model.logits
            else:
                output = model(data)
            check_labels(output, target, f"main target (eval, split={split})")
            loss_task = criterion(output, target)
            acc = accuracy(output, target)[0].item()
            tot_acc.update(acc, data.size(0))
            loss_phs = {i: 0 for i in range(len(phs))}
            for i, ph in enumerate(phs):
                ph_output = ph()
                check_labels(ph_output, private_label, f"private_label PH{i} (epoch_model_eval, split={split})")
                loss_phs[i] += criterion(ph_output, private_label)
                tot_loss_phs[i].update(loss_phs[i].item(), data.size(0))
                tot_acc_phs[i].update(
                    accuracy(ph_output, private_label)[0].item(), data.size(0)
                )
                nb_priv = getattr(args, "private_num_classes", 2)
                mi_ph[i] = compute_MI(private_label, ph, nb_classes=nb_priv, args=args)
                tot_mi[i].update(mi_ph[i].item(), data.size(0))
            loss = (
                compute_joint_loss(loss_task, mi_ph, args, loss_type="irene")
                if minimize_MI
                else loss_task
            )
            tot_loss.update(loss.item(), data.size(0))
            if printing:
                tk.set_postfix(
                    {
                        "loss": round(tot_loss.avg, 4),
                        "acc": round(tot_acc.avg, 4),
                        "phs_a": {
                            i: round(tot_acc_phs[i].avg, 3) for i in range(len(phs))
                        },
                        "phs_l": {
                            i: round(tot_loss_phs[i].avg, 3) for i in range(len(phs))
                        },
                        "mi": {i: round(tot_mi[i].avg, 4) for i in range(len(phs))},
                    }
                )
        val_log = {
            "loss": tot_loss.avg,
            "acc": tot_acc.avg,
            "ph": {
                i: {
                    "acc": tot_acc_phs[i].avg,
                    "loss": tot_loss_phs[i].avg,
                    "mi": tot_mi[i].avg,
                }
                for i in range(len(phs))
            },
        }
    return val_log


def get_thresholds(prop_to_go, prop_to_keep, criterion, max_accuracy_model, block):
    if criterion == "proportion":
        threshold_to_go = prop_to_go * max_accuracy_model
        threshold_to_keep = prop_to_keep * max_accuracy_model
    elif criterion == "absolute":
        threshold_to_go = prop_to_go
        threshold_to_keep = prop_to_keep
    elif criterion == "proportion_block":
        threshold_to_go = (prop_to_go ** (block + 1)) * max_accuracy_model
        threshold_to_keep = (prop_to_keep ** (block + 1)) * max_accuracy_model
    print(f"Threshold to go: {threshold_to_go}, threshold to keep: {threshold_to_keep}")
    return threshold_to_go, threshold_to_keep


def generate_sparsity_candidates(num_channels):
    if num_channels <= 0:
        return [0.0]
    if num_channels == 1:
        return [0.0, 1.0]
    values = np.linspace(0.0, 1.0, num_channels)
    return [float(np.round(v, 6)) for v in values]


def update_sparsities(
        sparsities,
        sparsity_list,
        block,
        sparsity,
        threshold_to_go,
        max_accuracy_model,
        best_val_acc_sparsity,
        best_val_acc_block,
        no_improvement_counter,
        best_previous_sparsity=0,
        available_sparsities=None,
):
    if no_improvement_counter > 3 and available_sparsities is not None:
        lower_bound = min(best_previous_sparsity, sparsity)
        upper_bound = max(best_previous_sparsity, sparsity)
        narrowed = [
            value
            for value in available_sparsities
            if lower_bound <= value <= upper_bound
        ]
        if narrowed:
            sparsity_list = narrowed
            print(f"No improvement for 3 epochs")
    if best_val_acc_sparsity < threshold_to_go:
        print(
            f"Block {block} with sparsity {sparsity} did not reach {threshold_to_go}, trying a lower sparsity"
        )
        sparsity_list = sparsity_list[: len(sparsity_list) // 2]
    else:
        pct = (best_val_acc_sparsity * 100.0) / max_accuracy_model if max_accuracy_model > 0 else 0.0
        print(
            f"Block {block} with sparsity {sparsity} reached {best_val_acc_sparsity} = {pct}% of the baseline, trying a higher sparsity"
        )
        sparsity_list = sparsity_list[len(sparsity_list) // 2:]
    if len(sparsity_list) > 1:
        print(f"New sparsity list from {sparsity_list[0]} to {sparsity_list[-1]}")
    return sparsities, sparsity_list


def adapt_and_test_phs(phs, dls, criterion, args, dir_name, nb_epochs_ph, ):
    ph_optimizers = [torch.optim.SGD(
        phs[i].parameters(),
        lr=args.lr_ph[0],
        momentum=args.mom_sgd,
        weight_decay=args.wd,
    ) for i in range(len(phs))]

    ph_gamma = compute_exponential_gamma(
        args.lr_ph[0], args.lr_ph[0] / 100.0, nb_epochs_ph
    )
    ph_schedulers = {
        i: torch.optim.lr_scheduler.ExponentialLR(
            ph_optimizers[i], gamma=ph_gamma
        )
        for i in range(len(phs))
    }
    best_val_losses = {i: np.inf for i in range(len(phs))}
    best_phs_path_temp = {}
    to_train = [i for i in range(len(phs))]
    for i in to_train:
        best_phs_path_temp[i] = os.path.join(dir_name, f"best_ph{i}_{args.model}.pth")
        torch.save(phs[i].classifier.state_dict(), best_phs_path_temp[i])
    for e in range(nb_epochs_ph):
        if len(to_train) == 0:
            break
        ph_training_log = epoch_ph_training(
            model, phs, dls, criterion, ph_optimizers, args,
            to_train=to_train, epoch_ph=e, epoch_ph_begin=nb_epochs_model
        )
        ph_eval_log = epoch_eval_ph(model, phs, dls, criterion, args, to_train=to_train)
        ph_test_log = epoch_eval_ph(model, phs, dls, criterion, args, to_train=to_train, split="test")
        wandb.log(
            {
                "val": ph_eval_log,
                "test": ph_test_log,
                "train": ph_training_log,
                "epoch_ph": e,
                f"ph": {f"{i}.lr": ph_optimizers[i].param_groups[0]["lr"]
                        for i in range(len(phs))},
                "no_improvement_epochs": {f"ph{i}": None for i in range(len(phs))},
            }
        )
        for i in to_train[:]:
            prev_loss = best_val_losses[i]
            current_loss = ph_eval_log["ph"][i]["loss"]
            ph_schedulers[i].step()
            print("Best val loss:", prev_loss, "current val loss:", current_loss)
            if current_loss < prev_loss:
                print(f"Saving best ph{i}: epoch ", e, "with loss", current_loss)
                best_val_losses[i] = current_loss
                best_phs_path_temp[i] = os.path.join(dir_name, f"best_ph{i}_{args.model}.pth")
                torch.save(phs[i].classifier.state_dict(), best_phs_path_temp[i])
    return phs, best_val_losses, best_phs_path_temp, to_train, ph_schedulers


def training_pre_pruning_model(
        model, phs, dls, criterion, args, dir_name, nb_epochs_model, minimize_MI=True, block=0,
):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.mom_sgd, weight_decay=args.wd
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=nb_epochs_model, eta_min=args.lr / 100, last_epoch=-1
    )
    ph_optimizers = [torch.optim.SGD(ph.parameters(), lr=args.lr_ph[0], momentum=args.mom_sgd, weight_decay=args.wd) for
                     i, ph in enumerate(phs)]
    best_val_loss = np.inf
    for e in range(nb_epochs_model):
        train_log = epoch_model_training(model, phs, dls, criterion, optimizer, ph_optimizers, e, args,
                                         minimize_MI=True, train_model=True, train_PH=[True for _ in range(len(phs))], )
        val_log = epoch_model_eval(model, phs, dls, criterion, args)
        test_log = epoch_model_eval(model, phs, dls, criterion, args, minimize_MI=False, split="test")
        log = {"train": train_log,
               "val": val_log,
               "test": test_log,
               "epoch": e,
               "lr": optimizer.param_groups[0]["lr"], }
        wandb.log(log | {"pre_pruning_Irene": log})
        scheduler.step()
        if val_log["loss"] < best_val_loss:
            best_val_loss = val_log["loss"]
            filename = (
                f"best_prepruned_block{block}.pth"
                if args.save_only_best_models
                else f"best_prepruned_block{block}_epoch{e}.pth"
            )
            best_path = os.path.join(dir_name, filename)
            torch.save(model.state_dict(), best_path)
            print(f"New best validation loss: {best_val_loss:.4f} at epoch {e}, saved to {best_path}")
    torch.save(model.state_dict(), best_path)
    path_to_prepruned_model = best_path
    return path_to_prepruned_model


def train_privacy_heads(
        model,
        phs,
        dls,
        criterion,
        args,
        dir_name,
        e_ph,
        lr_ph,
        init_from_paths=None,
        tag="",
):
    """Wrapper implementing TRAINPRIVACYHEADS as described in Algorithm 3."""

    if init_from_paths is not None:
        for i, path in enumerate(init_from_paths):
            checkpoint = torch.load(path, map_location=args.device, weights_only=True)
            try:
                phs[i].classifier.load_state_dict(checkpoint)
            except RuntimeError as e:
                print(
                    "[WARNING] Privacy head checkpoint has incompatible shapes; "
                    "loading only matching parameters."
                )
                print(f"           Details: {e}")
                if hasattr(phs[i].classifier, "load_compatible_state_dict"):
                    phs[i].classifier.load_compatible_state_dict(checkpoint)
                else:
                    # Fallback to strict=False if custom loader is unavailable
                    phs[i].classifier.load_state_dict(checkpoint, strict=False)

    # Algorithm 1: update the privacy heads only, without backpropagating into the encoder.
    model.eval()
    freeze_model_parameters(model)

    ph_optimizers = [
        torch.optim.SGD(
            phs[i].parameters(),
            lr=lr_ph,
            momentum=args.mom_sgd,
            weight_decay=args.wd,
        )
        for i in range(len(phs))
    ]

    ph_gamma = compute_exponential_gamma(lr_ph, lr_ph / 100.0, e_ph)
    ph_schedulers = [
        torch.optim.lr_scheduler.ExponentialLR(ph_optimizer, gamma=ph_gamma)
        for ph_optimizer in ph_optimizers
    ]

    best_val_losses = {i: np.inf for i in range(len(phs))}
    best_ph_paths = {}

    for i in range(len(phs)):
        best_ph_paths[i] = os.path.join(dir_name, f"{tag}_ph{i}_{args.model}.pth")
        torch.save(phs[i].classifier.state_dict(), best_ph_paths[i])

    for e in range(e_ph):
        ph_train_log = epoch_ph_training(
            model,
            phs,
            dls,
            criterion,
            ph_optimizers,
            args,
            to_train="all",
            epoch_ph=e,
            epoch_ph_begin=0,
        )
        ph_val_log = epoch_eval_ph(
            model,
            phs,
            dls,
            criterion,
            args,
            to_train="all",
            split="val",
        )

        if wandb.run is not None:
            wandb.log(
                {
                    f"{tag}/ph_train": ph_train_log,
                    f"{tag}/ph_val": ph_val_log,
                    f"{tag}/epoch_ph": e,
                }
            )

        for i in range(len(phs)):
            current_loss = ph_val_log["ph"][i]["loss"]
            if current_loss < best_val_losses[i]:
                best_val_losses[i] = current_loss
                torch.save(phs[i].classifier.state_dict(), best_ph_paths[i])

        for scheduler in ph_schedulers:
            scheduler.step()

    for i in range(len(phs)):
        phs[i].classifier.load_state_dict(
            torch.load(best_ph_paths[i], map_location=args.device, weights_only=True)
        )

    # Re-enable training of the whole model before Algorithm 2.
    unfreeze_model_parameters(model)

    ordered_paths = [best_ph_paths[i] for i in range(len(phs))]
    return phs, ordered_paths


def train_model_with_mi(
        model,
        phs,
        dls,
        criterion,
        args,
        dir_name,
        e_model,
        lr_model,
        gamma,
        tag="",
        train_ph=False,
):
    """Wrapper implementing TRAINMODELWITHMI with explicit MI control."""

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr_model, momentum=args.mom_sgd, weight_decay=args.wd
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=e_model, eta_min=lr_model / 100
    )

    ph_optimizers = [
        torch.optim.SGD(
            ph.parameters(),
            lr=args.lr_ph[0],
            momentum=args.mom_sgd,
            weight_decay=args.wd,
        )
        for ph in phs
    ]

    best_val_acc = -1.0
    best_model_path = os.path.join(dir_name, f"{tag}_best_model_with_mi.pth")
    train_PH_flags = [train_ph for _ in range(len(phs))]

    for e in range(e_model):
        train_log = epoch_model_training(
            model=model,
            phs=phs,
            dls=dls,
            criterion=criterion,
            optimizer=optimizer,
            ph_optimizers=ph_optimizers,
            epoch=e,
            args=args,
            minimize_MI=True,
            train_model=True,
            train_PH=train_PH_flags,
            gamma=gamma,
        )

        val_log = epoch_model_eval(
            model,
            phs,
            dls,
            criterion,
            args,
            minimize_MI=False,
            printing=False,
            split="val",
        )

        if wandb.run is not None:
            wandb.log(
                {
                    f"{tag}/train": train_log,
                    f"{tag}/val": val_log,
                    f"{tag}/epoch": e,
                    f"{tag}/lr": optimizer.param_groups[0]["lr"],
                }
            )

        scheduler.step()

        if val_log["acc"] > best_val_acc:
            best_val_acc = val_log["acc"]
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(
        torch.load(best_model_path, map_location=args.device, weights_only=True)
    )

    return model, best_model_path, best_val_acc


if __name__ == "__main__":
    group = "test"
    args = parse_args()
    project = args.projectName
    dls = {}
    dls["train"], dls["val"], dls["test"] = build_dataloaders(args)
    print_sensitive_attribute_distribution(dls)

    # Infer how many classes target and private labels have (e.g. 2 for CelebA, 10 for corrupted CIFAR)
    infer_private_num_classes(dls, args)
    infer_target_num_classes(dls, args)
    print(f"Private num classes inferred: {getattr(args, 'private_num_classes', 'N/A')}")
    print(f"Target num classes inferred: {getattr(args, 'target_num_classes', 'N/A')}")
    models_dir = args.save_path
    os.makedirs(models_dir, exist_ok=True)
    prefix = f"{args.projectName}_{args.model}_{args.dataset}_seed{args.seed}"
    print(prefix)
    folders = [
        f for f in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, f)) and f.startswith(prefix)
    ]
    print(folders)
    folders.sort()
    reload_folder = folders[-1] if folders else None
    if args.dataset.split("-")[0] == "celeba":
        beginning_model_file = "unbiased_best_raw_model"
        beginning_ph_file = "unbiased_best_raw_ph"
    else:
        beginning_model_file = "best_raw_model"
        beginning_ph_file = "best_raw_ph"
    reload_config = None
    reload_config_PHs = []
    if reload_folder is not None:
        reload_path = os.path.join(models_dir, reload_folder)
        model_files = [f for f in os.listdir(reload_path) if f.startswith(beginning_model_file)]
        epoch_numbers = [int(re.search(r'epoch(\d+)', f).group(1)) for f in model_files if re.search(r'epoch(\d+)', f)]
        if epoch_numbers:
            sorted_files = [f for _, f in sorted(zip(epoch_numbers, model_files))]
            reload_config = os.path.join(reload_path, sorted_files[-1])
            print("Last model file:", reload_config)
        for file in os.listdir(reload_path):
            if file.startswith(beginning_ph_file):
                reload_config_PHs.append(os.path.join(reload_path, file))
        reload_config_PHs.sort()
        print(f"Found {len(reload_config_PHs)} PH files: {reload_config_PHs}")
    else:
        print("No previous checkpoints found; starting fresh.")
    model = init_model(args, args.model)
    phs, bottleneck_layers = init_and_plug_phs(model, args, model_type=args.model, reload_config=None)
    if reload_config:
        print(f"Reloading model from {reload_config}")
        model.load_state_dict(torch.load(reload_config, map_location=args.device, weights_only=True))
    else:
        print("No model checkpoint to reload; using freshly initialized model.")
    handle_all(args, len(phs))
    time.sleep(10)
    args.nb_epochs_ph = args.nb_epochs_ph + [args.nb_epochs_ph[0]] * (
            len(phs) - len(args.nb_epochs_ph)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.mom_sgd, weight_decay=args.wd
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.sched_factor,
                                                           patience=args.patience, min_lr=args.min_lr,
                                                           threshold_mode="rel", threshold=args.threshold_sched,
                                                           )
    ph_optimizers = [torch.optim.SGD(ph.parameters(), lr=args.lr_ph[0], momentum=args.mom_sgd, weight_decay=args.wd) for
                     i, ph in enumerate(phs)]
    train_model = True
    train_PH = [True for i in range(len(phs))]
    epoch_ph = 0
    model_type = args.model
    prop_to_go = 0.95 ** (1 / len(bottleneck_layers))
    prop_to_keep = 0.95 ** (1 / len(bottleneck_layers))
    nb_epochs_ph = args.nb_epochs_ph[0] if isinstance(args.nb_epochs_ph, list) else args.nb_epochs_ph
    nb_epochs_model = args.refresh_epochs_model
    nb_epochs_model_pre_pruning = args.nb_epochs
    blocks = [i for i in range(len(bottleneck_layers))]

    print("Raw model")
    cur_date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    run_name = f"{args.projectName}_{args.model}_{args.dataset}_seed{args.seed}_{cur_date}"
    dir_name = os.path.join(models_dir, run_name)
    os.makedirs(dir_name, exist_ok=True)
    summary_path, stats_dir = init_experiment_tracking(dir_name, args)
    # Ensure CSV/log outputs have a dedicated destination
    args.output = stats_dir

    if args.wandb_log:
        if wandb.run is not None:
            wandb.finish()
        wandb.init(
            project=project,
            name=f"{run_name}_full_pipeline",
            group=group,
            config={
                "model": args.model,
                "dataset": args.dataset,
                "seed": args.seed,
                "gamma": args.gamma,
                "used_phs": args.used_phs,
                "Nsparsities": args.Nsparsities,
            },
            reinit=True,
        )
        log_full_config_to_wandb(args)

    raw_log = epoch_model_eval(
        model,
        phs,
        dls,
        criterion,
        args,
        minimize_MI=False,
        printing=False,
        split="val",
    )
    if wandb.run is not None:
        wandb.log({"raw_eval": raw_log})

    # =======================
    # PRETRAIN cycles:
    #   - 1 PH epoch (backbone frozen)
    #   - 1 backbone epoch with MI minimization
    # =======================
    pretrain_epochs = nb_epochs_model_pre_pruning
    best_pretrained_model_path = os.path.join(dir_name, "best_pretrained_model.pth")

    # Number of retraining epochs after each pruning step (same schedule as the pretrain)
    retrain_epochs_after_pruning = 10

    optimizer_model = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.mom_sgd, weight_decay=args.wd
    )

    lr_init = args.lr
    lr_final = lr_init / 100.0
    opt_gamma = (lr_final / lr_init) ** (1.0 / pretrain_epochs)

    print(f"[PRETRAIN MI] Using ExponentialLR with gamma={opt_gamma:.6f}")

    scheduler_model = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_model,
        gamma=opt_gamma,
    )

    # optimizers for the privacy heads during the alternating pretrain
    ph_optimizers_epoch = [
        torch.optim.SGD(
            ph.parameters(),
            lr=args.lr_ph[0],
            momentum=args.mom_sgd,
            weight_decay=args.wd,
        )
        for ph in phs
    ]

    ph_gamma_pretrain = compute_exponential_gamma(
        args.lr_ph[0], args.lr_ph[0] / 100.0, pretrain_epochs
    )
    ph_schedulers_epoch = [
        torch.optim.lr_scheduler.ExponentialLR(ph_opt, gamma=ph_gamma_pretrain)
        for ph_opt in ph_optimizers_epoch
    ]

    for e in range(pretrain_epochs):
        # Batch-wise MI pretraining; for EVERY batch:
        #   1) update the PHs on the batch
        #   2) compute MI on that batch
        #   3) update the backbone on the same batch minimizing MI
        train_log = epoch_model_training(
            model=model,
            phs=phs,
            dls=dls,
            criterion=criterion,
            optimizer=optimizer_model,
            ph_optimizers=ph_optimizers_epoch,
            epoch=e,
            args=args,
            minimize_MI=True,                           # use MI in the loss
            train_model=True,                           # update the backbone
            train_PH=[True for _ in range(len(phs))],   # update the PHs as well
        )

        val_log = epoch_model_eval(
            model,
            phs,
            dls,
            criterion,
            args,
            minimize_MI=False,
            printing=False,
            split="val",
        )

        if wandb.run is not None:
            wandb.log(
                {
                    "pretrain/train": train_log,
                    "pretrain/val": val_log,
                    "pretrain/epoch": e,
                    "pretrain/lr": optimizer_model.param_groups[0]["lr"],
                }
            )

        scheduler_model.step()

        for ph_scheduler in ph_schedulers_epoch:
            ph_scheduler.step()

    torch.save(model.state_dict(), best_pretrained_model_path)
    print(f"[PRETRAIN MI] Saved pretrained model to {best_pretrained_model_path}")

    # reload the best pretrained model (backbone trained with MI + PHs trained alternately)
    model.load_state_dict(
        torch.load(best_pretrained_model_path, map_location=args.device, weights_only=True)
    )
    pretrain_eval_log = epoch_model_eval(
        model,
        phs,
        dls,
        criterion,
        args,
        minimize_MI=False,
        printing=False,
        split="val",
    )
    vanilla_acc = pretrain_eval_log["acc"]
    if wandb.run is not None:
        wandb.log({"pretrain/final_eval": pretrain_eval_log})

    e_PH = nb_epochs_ph
    lr_PH = args.lr_ph[0]
    phs, vanilla_ph_paths = train_privacy_heads(
        model=model,
        phs=phs,
        dls=dls,
        criterion=criterion,
        args=args,
        dir_name=dir_name,
        e_ph=e_PH,
        lr_ph=lr_PH,
        init_from_paths=None,
        tag="vanilla",
    )

    # =======================
    # GLOBAL PRUNING
    #   - a single search for the total sparsity s
    # =======================

    # Number of blocks, used only to derive the global threshold (≈ 0.95 * vanilla_acc)
    B = len(bottleneck_layers)

    transformer_pruning = init_transformer_mlp_pruning(model, args.model)

    # Modules prunable globally (conv + linear) across the whole network
    all_prunable_modules = []
    module_to_name = {}
    if not transformer_pruning.enabled:
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                all_prunable_modules.append(module)
                module_to_name[module] = name

    # Candidate sparsity list, searched once globally
    Nsparsities = max(args.Nsparsities, 1)
    base_Slist = [i / Nsparsities for i in range(1, Nsparsities + 1)]

    # Global accuracy threshold, computed for a single "mega-block": prop_to_go is
    # 0.95 ** (1 / B), so raising it to the power B gives back ≈ 0.95 * vanilla_acc.
    global_prop_to_go = prop_to_go ** B
    global_prop_to_keep = prop_to_keep ** B
    T_acc_global, _ = get_thresholds(
        global_prop_to_go,
        global_prop_to_keep,
        criterion="proportion",
        max_accuracy_model=vanilla_acc,
        block=0,
    )

    print(f"[GLOBAL PRUNING] Accuracy threshold: {T_acc_global:.4f} (base acc = {vanilla_acc:.4f})")

    Slist = base_Slist.copy()
    PH_LAST_IDX = len(phs) - 1
    PRIV_THR = 0.65  # threshold on PH_last, expressed in the 0..1 scale

    def _acc_to_frac(a: float) -> float:
        return a / 100.0 if a > 1.0 else a

    current_ok_solution = None  # (s, task_acc, priv_last_acc_frac) — tracking only

    best_model_path = None      # best ACCEPTABLE model (highest task acc)
    best_task_acc = -1.0
    best_s = None
    best_priv_last_acc = None   # optional, for logging

    # fallback used when no candidate is acceptable:
    fallback_model_path = None  # model with the lowest priv_last_acc
    fallback_priv_last_acc = float("inf")
    fallback_task_acc = None
    fallback_s = None

    while len(Slist) > 1:
        # pick the candidate sparsity (bisection-style search)
        j = int(np.ceil(len(Slist) / 2.0)) - 1
        s = Slist[j]
        print(f"[GLOBAL PRUNING] Testing global sparsity s = {s:.6f}")

        # 1) drop any leftover pruning parameters
        #    (ALWAYS before loading the state_dict)
        if transformer_pruning.enabled:
            transformer_pruning.clear_pruning()
        else:
            for module in all_prunable_modules:
                if hasattr(module, "weight_orig"):
                    prune.remove(module, "weight")

        # 2) ALWAYS reload the pretrained (unpruned) model
        model.load_state_dict(
            torch.load(best_pretrained_model_path, map_location=args.device, weights_only=True)
        )

        if transformer_pruning.enabled:
            total_channels, prune_count = transformer_pruning.prune(s)
        else:
            # 3) global structured pruning: the whole network treated as a SINGLE block;
            #    compute an L1 metric per channel and keep explicit track of the indices
            global_channel_metrics = []

            for module in all_prunable_modules:
                weight = module.weight.detach()
                for channel_idx in range(weight.shape[0]):
                    # Normalize the L1 sum by the number of elements in the channel, so
                    # channels with smaller kernels are not penalized against larger ones.
                    channel_weights = weight[channel_idx]
                    magnitude = channel_weights.abs().sum().item() / channel_weights.numel()
                    global_channel_metrics.append((magnitude, module, channel_idx))

            total_channels = len(global_channel_metrics)
            prune_count = int(np.floor(s * total_channels))

            if prune_count > 0:
                global_channel_metrics.sort(key=lambda x: x[0])
                channels_to_prune = defaultdict(list)
                pruned_channels = []

                for _, module, channel_idx in global_channel_metrics[:prune_count]:
                    channels_to_prune[module].append(channel_idx)
                    pruned_channels.append((module_to_name.get(module, "unknown"), channel_idx))

                for module, channel_indices in channels_to_prune.items():
                    mask = torch.ones_like(module.weight)
                    for channel_idx in channel_indices:
                        mask[channel_idx] = 0
                    prune.custom_from_mask(module, name="weight", mask=mask)

        # ============================================
        # POST-PRUNING RETRAINING
        # ============================================

        # Reset the privacy heads to the "vanilla" weights before evaluating/retraining
        for i, path in enumerate(vanilla_ph_paths):
            phs[i].classifier.load_state_dict(
                torch.load(path, map_location=args.device, weights_only=True)
            )

        if args.pruning_retraining != "retrain":
            raise ValueError("Questo esperimento richiede retraining post-pruning (pruning_retraining='retrain').")

        # Optimizer and scheduler for the pruned model (as in the pretrain, but for 10 epochs)
        optimizer_retrain = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.mom_sgd, weight_decay=args.wd
        )
        lr_init_retrain = args.lr
        lr_final_retrain = lr_init_retrain / 100.0
        opt_gamma_retrain = (lr_final_retrain / lr_init_retrain) ** (1.0 / retrain_epochs_after_pruning)
        scheduler_retrain = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_retrain,
            gamma=opt_gamma_retrain,
        )

        # Optimizers and schedulers for the PHs during the retraining
        ph_optimizers_retrain = [
            torch.optim.SGD(
                ph.parameters(),
                lr=args.lr_ph[0],
                momentum=args.mom_sgd,
                weight_decay=args.wd,
            )
            for ph in phs
        ]
        ph_gamma_retrain = compute_exponential_gamma(
            args.lr_ph[0], args.lr_ph[0] / 100.0, retrain_epochs_after_pruning
        )
        ph_schedulers_retrain = [
            torch.optim.lr_scheduler.ExponentialLR(ph_opt, gamma=ph_gamma_retrain)
            for ph_opt in ph_optimizers_retrain
        ]

        # 10 pretrain-style retraining epochs: within each batch,
        # first update the PHs, then update the backbone with MI.
        for e_retrain in range(retrain_epochs_after_pruning):
            train_log = epoch_model_training(
                model=model,
                phs=phs,
                dls=dls,
                criterion=criterion,
                optimizer=optimizer_retrain,
                ph_optimizers=ph_optimizers_retrain,
                epoch=e_retrain,
                args=args,
                minimize_MI=True,                           # use MI in the loss
                train_model=True,                           # update the backbone
                train_PH=[True for _ in range(len(phs))],   # update the PHs as well
                printing=False,
            )

            scheduler_retrain.step()
            for ph_sched in ph_schedulers_retrain:
                ph_sched.step()

        # Validation evaluation after the pretrain-style retraining
        val_log = epoch_model_eval(
            model,
            phs,
            dls,
            criterion,
            args,
            minimize_MI=False,
            printing=False,
            split="val",
        )
        task_acc = val_log["acc"]

        # accuracy of the LAST privacy head only
        priv_last_acc_raw = val_log["ph"][PH_LAST_IDX]["acc"]
        priv_last_acc = _acc_to_frac(priv_last_acc_raw)

        is_ok = (priv_last_acc <= PRIV_THR)

        print(
            f"[GLOBAL PRUNING] s={s:.6f}  task_acc={task_acc:.4f}  "
            f"priv_last_acc={priv_last_acc:.4f}  ok={is_ok}"
        )

        if wandb.run is not None:
            wandb.log({
                "global_pruning/candidate/sparsity": s,
                "global_pruning/candidate/task_acc": task_acc,
                "global_pruning/candidate/priv_last_acc": priv_last_acc,
                "global_pruning/candidate/is_ok": int(is_ok),
            })

        # ALWAYS refresh the fallback (only needed if no acceptable candidate exists)
        if priv_last_acc < fallback_priv_last_acc:
            fallback_priv_last_acc = priv_last_acc
            fallback_task_acc = task_acc
            fallback_s = s
            fallback_model_path = os.path.join(dir_name, f"fallback_min_priv_s_{s:.6f}.pth")
            torch.save(model.state_dict(), fallback_model_path)

        m = int(np.ceil(len(Slist) / 2.0))

        if not is_ok:
            # priv_last_acc > 0.65 => prune more => keep the right half
            Slist = Slist[m:]
        else:
            # acceptable => store current_ok and try less pruning => keep the left half
            current_ok_solution = (s, task_acc, priv_last_acc)
            Slist = Slist[:m]

            # best model = highest task accuracy among the ACCEPTABLE ones
            if task_acc > best_task_acc:
                best_task_acc = task_acc
                best_s = s
                best_priv_last_acc = priv_last_acc
                best_model_path = os.path.join(dir_name, f"best_ok_task_s_{s:.6f}.pth")
                torch.save(model.state_dict(), best_model_path)

    if best_model_path is None:
        print("[GLOBAL PRUNING] Nessuna sparsity è risultata accettabile (priv_last_acc <= 0.65).")
        # fallback: lowest priv_last_acc
        if fallback_model_path is None:
            # extreme edge case: no candidate was ever evaluated
            best_model_path = best_pretrained_model_path
            best_s = 0.0
            best_task_acc = vanilla_acc
            best_priv_last_acc = None
        else:
            best_model_path = fallback_model_path
            best_s = fallback_s
            best_task_acc = fallback_task_acc
            best_priv_last_acc = fallback_priv_last_acc
    else:
        print("[GLOBAL PRUNING] Trovato modello accettabile migliore per task accuracy.")

    # For compatibility with the logging/export logic, build a fake global "block 0"
    block_tracking = {
        0: {
            "sparsity": float(best_s) if best_s is not None else 0.0,
            "val_acc": float(best_task_acc) if best_task_acc is not None else -1.0,
            "priv_last_acc": float(best_priv_last_acc) if best_priv_last_acc is not None else None,
        }
    }
    log_block_summary(summary_path, 0, block_tracking[0]["sparsity"], block_tracking[0]["val_acc"])
    if wandb.run is not None:
        wandb.log({
            "global_pruning/summary": {
                "sparsity": block_tracking[0]["sparsity"],
                "val_acc": block_tracking[0]["val_acc"],
                "priv_last_acc": block_tracking[0]["priv_last_acc"],
            }
        })

    def _fold_pruning_state(state_dict):
        """Drop the pruning reparametrization (weight_orig/mask) by folding the mask into the weight."""
        folded_state = {}
        for key, tensor in state_dict.items():
            if key.endswith("weight_orig"):
                base_key = key[: -len("weight_orig")] + "weight"
                mask_key = base_key + "_mask"
                mask = state_dict.get(mask_key)
                folded_state[base_key] = tensor if mask is None else tensor * mask
            elif key.endswith("weight_mask"):
                continue  # already folded into weight_orig
            else:
                folded_state[key] = tensor
        return folded_state

    def _remove_pruning_reparam(modules):
        for m in modules:
            if hasattr(m, "weight_orig"):
                prune.remove(m, "weight")

    def _sync_pruning_reparam_for_load(state_dict, transformer_pruning, model, all_prunable_modules):
        if transformer_pruning.enabled:
            return state_dict  # handled by its own pruning tool

        needs_reparam = any(k.endswith("weight_orig") for k in state_dict.keys())
        # Widen the set of modules to inspect, including any leftover reparametrizations
        candidate_modules = list(all_prunable_modules)
        for module in model.modules():
            if hasattr(module, "weight_orig") and module not in candidate_modules:
                candidate_modules.append(module)

        has_reparam = any(hasattr(m, "weight_orig") for m in candidate_modules)

        # If the checkpoint holds weight_orig/weight_mask but the current model has no
        # active reparametrization, add it in place to align the keys.
        if needs_reparam and not has_reparam:
            target_modules = candidate_modules
            if not target_modules:
                target_modules = [
                    m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))
                ]
            for m in target_modules:
                prune.identity(m, "weight")
            has_reparam = any(hasattr(m, "weight_orig") for m in target_modules)

        # If the checkpoint does NOT hold the reparametrization but the model does, remove
        # it to avoid key conflicts while loading.
        if (not needs_reparam) and has_reparam:
            _remove_pruning_reparam(candidate_modules)
            has_reparam = False

        # If after the previous attempts the state still carries the reparametrization but
        # the model exposes no weight_orig, fold the mask into "weight" so the checkpoint
        # can be loaded anyway.
        if needs_reparam and not has_reparam:
            state_dict = _fold_pruning_state(state_dict)

        return state_dict

    state = torch.load(best_model_path, map_location=args.device, weights_only=True)
    state = _sync_pruning_reparam_for_load(state, transformer_pruning, model, all_prunable_modules)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        # Robust adaptation in case of residual mismatches between state and model.
        print(f"[WARNING] Reload with pruning params failed ({exc}); folding masks and retrying.")
        state = _fold_pruning_state(state)
        _remove_pruning_reparam([m for m in model.modules() if hasattr(m, "weight_orig")])
        model.load_state_dict(state)

    from pruning import eval_sparsity, eval_channel_sparsity

    global_s, block_s = eval_sparsity(model, args)
    print("\n[DEBUG] Global weight sparsity:", global_s)
    print("[DEBUG] Per-block sparsity:", block_s)
    if wandb.run is not None:
        wandb.log({
            "sparsity/global": global_s,
            "sparsity/per_block": block_s,
        })

    channel_stats, channel_summary = eval_channel_sparsity(
        model,
        args=args,
        sort_by="sparsity",
        descending=True,
        top_k=50,   # avoid huge logs
        csv_path=os.path.join(args.output, "channel_sparsity.csv"),
        wandb_prefix="sparsity/channel"
    )

    total_channels = channel_summary["total_channels"]
    pruned_channels = channel_summary["pruned_channels"]
    percent_pruned = channel_summary["relative_sparsity"] * 100.0

    print(
        f"[SPARSITY] Canali prunati: {pruned_channels}/{total_channels} "
        f"({percent_pruned:.2f}% )"
    )

    channel_summary_path = os.path.join(args.output, "channel_pruning_summary.txt")
    with open(channel_summary_path, "a", encoding="utf-8") as f:
        f.write(
            f"Pruned channels: {pruned_channels}/{total_channels} "
            f"({percent_pruned:.2f}%)\n"
        )

    if wandb.run is not None:
        wandb.log(
            {
                "sparsity/channel_pruned_percent": percent_pruned,
                "sparsity/channel_pruned_count": pruned_channels,
                "sparsity/channel_total": total_channels,
            }
        )

    # rebuild / re-attach the privacy heads on the final PRUNED model
    final_eval_phs, _ = init_and_plug_phs(model, args, model_type=args.model)

    # TRAIN ONLY THE PRIVACY HEADS on the final pruned model
    final_eval_phs, _ = train_privacy_heads(
        model=model,                 # final pruned backbone
        phs=final_eval_phs,
        dls=dls,
        criterion=criterion,
        args=args,
        dir_name=dir_name,
        e_ph=e_PH,
        lr_ph=lr_PH,
        init_from_paths=vanilla_ph_paths,  # or None to start from scratch
        tag="final_eval",
    )

    # FINAL EVALUATION on train/val/test of the pruned model + retrained PHs
    eval_logs = {}
    split_labels = [
        ("train", "Training set"),
        ("val", "Evaluation set"),
        ("test", "Test set"),
    ]
    for split_name, label in split_labels:
        eval_logs[label] = epoch_model_eval(
            model,               # the same pruned 'model'
            final_eval_phs,
            dls,
            criterion,
            args,
            minimize_MI=False,
            printing=False,
            split=split_name,
        )

    for split_key, split_log in eval_logs.items():
        log_split_metrics(f"final/{split_key.lower().replace(' ', '_')}", split_log)

    final_ph_idx = len(final_eval_phs) - 1 if len(final_eval_phs) > 0 else None
    log_final_wandb_summary(
        eval_logs=eval_logs,
        last_ph_idx=final_ph_idx,
        global_sparsity=global_s,
        channel_relative_sparsity=channel_summary["relative_sparsity"],
    )

    save_all_stats(stats_dir, block_tracking, eval_logs)
    log_final_summary(summary_path, best_model_path, block_tracking, eval_logs)
    if wandb.run is not None:
        wandb.finish()
