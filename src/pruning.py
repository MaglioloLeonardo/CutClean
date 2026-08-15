from __future__ import print_function
import csv
import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import wandb
from datasets.celebA import *
from irene.utilities import *
from irene.core import *


def _module_belongs_to_block(name, block_idx, bottleneck_layers, args):
    """Return True if the module identified by *name* belongs to the block."""

    layer_name = bottleneck_layers[block_idx][0]

    if args.model in ["resnet18", "resnet18_noskip", "vit", "vit_b", "tiny_vit"]:
        return layer_name in name

    if "vgg" in args.model and "features." in name:
        if block_idx == 0:
            prev_block = "features.0"
        else:
            prev_block = bottleneck_layers[block_idx - 1][0]

        start_idx = int(prev_block.split(".")[1])
        end_idx = int(layer_name.split(".")[1])

        for k in range(start_idx, end_idx + 1):
            if name == f"features.{k}" or name.startswith(f"features.{k}."):
                return True

    return False


def get_block_channel_counts(model, args, bottleneck_layers):
    """Compute the number of prunable channels for each block."""

    block_channel_counts = {}

    for block_idx in range(len(bottleneck_layers)):
        channels = []

        for name, module in model.named_modules():
            if not (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)):
                continue

            if not _module_belongs_to_block(name, block_idx, bottleneck_layers, args):
                continue

            if hasattr(module, "out_channels"):
                channels.append(module.out_channels)
            elif hasattr(module, "out_features"):
                channels.append(module.out_features)

        block_channel_counts[block_idx] = min(channels) if channels else 0

    return block_channel_counts


def prune_model(model, args, amount, layers_to_prune=None, bottleneck_layers=None, printing=True):
    """
    Prune model's specific blocks with enhanced global structured pruning capabilities.

    Args:
        model: The neural network model
        args: Pruning configuration arguments
        layers_to_prune: Specific blocks/layers to prune
        amount: Per-block pruning amount (optional)

    Returns:
        dict: Detailed pruning statistics
    """
    if printing:
        print("\n--- Detailed Pruning Process ---")
        print(f"Pruning Method: {args.pruning_method}")
        print(f"Magnitude Type: {args.pruning_criterion}")
        print(f"Pruning Amount: {amount}")

    if amount is None:
        amount = args.pruning_amount
    amount = {l: amount for l in layers_to_prune if type(amount) == float}

    pruning_details = {}

    for i, block in enumerate(layers_to_prune):
        print(i)
        if printing:
            print(f"Block {block+1}")
        block_layers = []
        block_magnitudes = []
        layer_name = bottleneck_layers[block][0]
        # Collect layers and magnitudes for this block
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if args.model in ["resnet18", "resnet18_noskip", "vit", "vit_b", "tiny_vit"]:
                    add_module = layer_name in name
                elif "vgg" in args.model:
                    add_module = False
                    if "features." in name:
                        if i == 0:
                            prev_block = "features.0"
                        else:
                            prev_block = bottleneck_layers[i - 1][0]
                        for k in range(
                            int(prev_block.split(".")[1]), int(layer_name.split(".")[1]) + 1
                        ):
                            # Ensure exact match, not partial (e.g. prevent features.10 when looking for features.1)
                            if name == f"features.{k}" or name.startswith(f"features.{k}."):
                                add_module = True
                                break
                else:
                    add_module = False
                if add_module:
                    block_layers.append((module, "weight"))

                    if args.pruning_criterion == "gradient":
                        magnitude = (
                            torch.abs(module.weight.grad)
                            if module.weight.grad is not None
                            else torch.zeros_like(module.weight)
                        )
                    elif args.pruning_criterion == "gradient+magnitude":
                        magnitude = (
                            torch.abs(module.weight.grad) * torch.abs(module.weight)
                            if module.weight.grad is not None
                            else torch.abs(module.weight)
                        )
                    else:  # weight magnitude
                        magnitude = torch.abs(module.weight)
                    block_magnitudes.append(magnitude)

        block_key = f"block_{block+1}"
        block_amount = amount[block]
        if printing:
            print(f"Pruning amount/sparsity: {block_amount}")

        if args.pruning_method == "unstructured":
            for (module, _), magnitude in zip(block_layers, block_magnitudes):

                prune.global_unstructured(
                    [(module, "weight")],
                    pruning_method=prune.L1Unstructured,
                    amount=block_amount,
                )

        elif "global_structured" in args.pruning_method:
            # Global structured pruning for the entire block
            module_magnitudes = []
            total_block_magnitude = 0.0

            for (module, _), magnitude in zip(block_layers, block_magnitudes):
                if printing:
                    print("  Module:", module)
                    print("     Sum:", torch.sum(magnitude).item())
                    print("     Mean: ", torch.mean(magnitude).item())
                    print(magnitude.size())
                if "normalized" in args.pruning_method:
                    magnitude_norm = magnitude / torch.norm(magnitude).item()
                    magnitude_minmax = (magnitude - torch.min(magnitude)) / (
                        torch.max(magnitude) - torch.min(magnitude)
                    )
                module_magnitude = torch.mean(magnitude_norm, dim=(0, 1))
                if printing:
                    print("     Norm: ", module_magnitude)
                module_magnitude = torch.mean(magnitude_minmax, dim=(0, 1))
                if printing:
                    print("     Min/Max: ", module_magnitude)
                total_block_magnitude += module_magnitude
                module_magnitudes.append((module, module_magnitude))

        elif args.pruning_method == "local_structured":
            # Local structured pruning for the entire block
            for (module, _), magnitude in zip(block_layers, block_magnitudes):
                prune.ln_structured(
                    module,
                    name="weight",
                    amount=block_amount,
                    n=1,
                    dim=0,
                )

        # Detailed block pruning statistics
        pruning_details[block_key] = {
            "layers": [],
            "total_zero_weights": 0,
            "total_weights": 0,
            "target_sparsity": block_amount,
        }

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if args.model in ["resnet18", "resnet18_noskip", "vit", "vit_b", "tiny_vit"]:
                    add_module = layer_name in name
                elif "vgg" in args.model:
                    add_module = False
                    if "features." in name:
                        if i == 0:
                            prev_block = "features.0"
                        else:
                            prev_block = bottleneck_layers[i - 1][0]
                        for k in range(
                            int(prev_block.split(".")[1]), int(layer_name.split(".")[1]) + 1
                        ):
                            # Ensure exact match, not partial (e.g. prevent features.10 when looking for features.1)
                            if name == f"features.{k}" or name.startswith(f"features.{k}."):
                                add_module = True
                                break
                else:
                    add_module = False
                if add_module:
                    pruned_weights = module.weight.detach()
                    zero_weights = torch.sum(pruned_weights == 0).item()
                    total_weights = torch.numel(pruned_weights)

                    pruning_details[block_key]["layers"].append(
                        {
                            "name": name,
                            "zero_weights": zero_weights,
                            "total_weights": total_weights,
                            "sparsity": zero_weights / total_weights,
                        }
                    )
                    pruning_details[block_key]["total_zero_weights"] += zero_weights
                    pruning_details[block_key]["total_weights"] += total_weights
                    if printing:
                        print(f"  Layer {name}: {zero_weights / total_weights:.6f} sparsity")

        block_sparsity = pruning_details[block_key]["total_zero_weights"] / pruning_details[block_key][
            "total_weights"
        ]
        if printing:
            print(f"Block {block+1} actual sparsity: {block_sparsity:.6f}")
            print(f"Block {block+1} target sparsity: {block_amount:.6f}")

    return pruning_details


def test_pruning_robustness(model, args, layers_to_prune=None, amount=None):
    """
    Test the robustness of pruning by comparing model performance before and after pruning.

    Args:
        model: The neural network model
        args: Pruning configuration arguments
        layers_to_prune: Specific layers/blocks to prune
        amount: Pruning amount for each block

    Returns:
        dict: Pruning test results including performance metrics
    """
    import copy

    original_model = copy.deepcopy(model)

    initial_global_sparsity, initial_block_sparsity = eval_sparsity(original_model)

    print("\n--- Initial Model Sparsity ---")
    print(f"Global Sparsity: {initial_global_sparsity}")
    print("Block Sparsity:", initial_block_sparsity)

    pruning_details = prune_model(model, args, layers_to_prune, amount)

    final_global_sparsity, final_block_sparsity = eval_sparsity(model)

    print("\n--- Final Model Sparsity ---")
    print(f"Global Sparsity: {final_global_sparsity}")
    print("Block Sparsity:", final_block_sparsity)

    return {
        "initial_global_sparsity": initial_global_sparsity,
        "final_global_sparsity": final_global_sparsity,
        "initial_block_sparsity": initial_block_sparsity,
        "final_block_sparsity": final_block_sparsity,
        "pruning_details": pruning_details,
    }


def eval_sparsity(model, args=None):
    """
    Evaluate the sparsity of the model, both globally and per-block.

    Args:
        model: The neural network model
        args: Model configuration arguments to identify model type

    Returns:
        tuple: (global_sparsity, block_sparsity_dict)
    """
    zero_weight = 0
    whole_weight = 0

    layer_sparsity = {}
    block_sparsity = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weights = module.weight.detach()

            zeros = torch.sum(weights == 0).item()
            total = torch.numel(weights)

            sparsity = zeros / total if total > 0 else 0
            layer_sparsity[name] = sparsity

            zero_weight += zeros
            whole_weight += total

            # Extract block information based on model type
            if args and args.model in ["resnet18", "resnet18_noskip", "vit", "vit_b"]:
                # For ResNet-style models using layer1, layer2, etc.
                for block_name in ["layer1", "layer2", "layer3", "layer4"]:
                    if block_name in name:
                        if block_name not in block_sparsity:
                            block_sparsity[block_name] = {"zeros": 0, "total": 0, "layers": []}
                        block_sparsity[block_name]["zeros"] += zeros
                        block_sparsity[block_name]["total"] += total
                        block_sparsity[block_name]["layers"].append(name)
            elif args and "vgg" in args.model:
                # For VGG-style models using features
                if "features." in name:
                    block_idx = name.split(".")[1].split(".")[0]  # Extract feature block number
                    block_key = f"features_{block_idx}"
                    if block_key not in block_sparsity:
                        block_sparsity[block_key] = {"zeros": 0, "total": 0, "layers": []}
                    block_sparsity[block_key]["zeros"] += zeros
                    block_sparsity[block_key]["total"] += total
                    block_sparsity[block_key]["layers"].append(name)
            else:
                # Default fallback to detect blocks using common naming patterns
                for block_name in ["layer", "block", "features"]:
                    if block_name in name:
                        parts = name.split(".")
                        for i, part in enumerate(parts):
                            if block_name in part:
                                block_key = ".".join(parts[: i + 1])
                                if block_key not in block_sparsity:
                                    block_sparsity[block_key] = {"zeros": 0, "total": 0, "layers": []}
                                block_sparsity[block_key]["zeros"] += zeros
                                block_sparsity[block_key]["total"] += total
                                block_sparsity[block_key]["layers"].append(name)
                                break

    global_sparsity = zero_weight / whole_weight if whole_weight > 0 else 0

    block_sparsity_dict = {}
    for block_name, data in block_sparsity.items():
        if data["total"] > 0:
            block_sparsity_dict[block_name] = round(data["zeros"] / data["total"], 10)

    return global_sparsity, block_sparsity_dict


def eval_channel_sparsity(
    model,
    args=None,
    sort_by="sparsity",
    descending=True,
    top_k=None,
    csv_path=None,
    wandb_prefix="channel_sparsity",
):
    """
    Compute and print per-channel sparsity over ALL Conv2d / Linear layers.

    Returns:
        tuple:
            - list of dicts with per-channel sparsity information.
            - dict with the global summary (total/pruned/relative_sparsity/mean_sparsity).
    """

    channel_stats = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.detach()
            w_flat = w.view(w.size(0), -1)
            total_i = w_flat.size(1)

            zeros_per_channel = torch.sum(w_flat == 0, dim=1).cpu().tolist()

            for i, zeros_i in enumerate(zeros_per_channel):
                sparsity_i = zeros_i / total_i if total_i > 0 else 0
                channel_stats.append(
                    {
                        "layer": name,
                        "channel": i,
                        "zeros": zeros_i,
                        "total": total_i,
                        "sparsity": sparsity_i,
                    }
                )

    if sort_by == "layer":
        channel_stats.sort(key=lambda x: (x["layer"], x["channel"]), reverse=descending)
    else:
        channel_stats.sort(key=lambda x: x.get(sort_by, 0), reverse=descending)

    print("\n=== Channel-wise sparsity (global) ===")
    display_limit = top_k if top_k is not None else len(channel_stats)
    for entry in channel_stats[:display_limit]:
        print(
            f"Layer {entry['layer']} | out_ch={entry['channel']} | sparsity={entry['sparsity']:.6f}"
            f" ({entry['zeros']}/{entry['total']})"
        )

    total_channels = len(channel_stats)
    fully_zero_channels = sum(1 for entry in channel_stats if entry["sparsity"] == 1.0)
    mean_sparsity = sum(entry["sparsity"] for entry in channel_stats) / total_channels if total_channels > 0 else 0
    relative_sparsity = fully_zero_channels / total_channels if total_channels > 0 else 0

    print(f"Totale canali: {total_channels}")
    print(f"Canali completamente azzerati: {fully_zero_channels}")
    print(f"Sparsity media per canale: {mean_sparsity:.6f}")
    print(
        "Relative sparsity (channel-level masks pruned): "
        f"{relative_sparsity * 100:.2f}%"
    )

    if csv_path is not None:
        file_exists = os.path.isfile(csv_path)
        fieldnames = ["layer", "channel", "zeros", "total", "sparsity"]
        with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(channel_stats)

    if wandb.run is not None:
        log_limit = top_k if top_k is not None else 200
        for entry in channel_stats[:log_limit]:
            wandb.log(
                {
                    f"{wandb_prefix}/layer": entry["layer"],
                    f"{wandb_prefix}/channel": entry["channel"],
                    f"{wandb_prefix}/sparsity": entry["sparsity"],
                }
            )

        wandb.log(
            {
                f"{wandb_prefix}/relative_sparsity": relative_sparsity,
                f"{wandb_prefix}/pruned_channels": fully_zero_channels,
                f"{wandb_prefix}/total_channels": total_channels,
                f"{wandb_prefix}/mean_sparsity": mean_sparsity,
            }
        )

    summary = {
        "total_channels": total_channels,
        "pruned_channels": fully_zero_channels,
        "relative_sparsity": relative_sparsity,
        "mean_sparsity": mean_sparsity,
    }

    return channel_stats, summary
