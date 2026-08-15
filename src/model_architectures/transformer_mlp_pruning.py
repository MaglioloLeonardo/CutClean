"""Transformer MLP-only pruning aligned with the existing ViT methodology.

This module extends the exact pruning strategy used for ViT MLPs to other
Transformer architectures by changing only how (fc1, fc2) pairs are collected
from Transformer blocks. The pruning unit, scoring, masking, and API remain
unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def _iter_transformer_blocks(model: nn.Module) -> Iterable[nn.Module]:
    """Yield Transformer blocks from common container attributes."""

    for attr in ("blocks", "layers"):
        blocks = getattr(model, attr, None)
        if blocks is not None:
            for block in blocks:
                yield block

    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        blocks = getattr(encoder, "layers", None)
        if blocks is not None:
            for block in blocks:
                yield block


def _linear_pairs_from_mlp(mlp: nn.Module) -> List[Tuple[nn.Linear, nn.Linear]]:
    """Return valid (fc1, fc2) pairs from an MLP module."""

    named_pairs: List[Tuple[nn.Linear, nn.Linear]] = []
    name_candidates = [("fc1", "fc2"), ("linear1", "linear2"), ("w1", "w2")]
    for name1, name2 in name_candidates:
        fc1 = getattr(mlp, name1, None)
        fc2 = getattr(mlp, name2, None)
        if isinstance(fc1, nn.Linear) and isinstance(fc2, nn.Linear):
            if fc1.out_features == fc2.in_features:
                named_pairs.append((fc1, fc2))
    if named_pairs:
        return named_pairs

    linears: List[nn.Linear] = []
    for module in mlp.modules():
        if module is mlp:
            continue
        if isinstance(module, nn.Linear):
            linears.append(module)

    pairs: List[Tuple[nn.Linear, nn.Linear]] = []
    for first, second in zip(linears, linears[1:]):
        if first.out_features == second.in_features:
            pairs.append((first, second))
    return pairs


def collect_transformer_mlp_pairs(model: nn.Module) -> List[Tuple[nn.Linear, nn.Linear]]:
    """Collect all (fc1, fc2) MLP pairs from Transformer blocks.

    Valid MLPs are searched inside Transformer blocks exposed through
    ``model.blocks``, ``model.layers``, or ``model.encoder.layers`` and must
    contain two consecutive ``nn.Linear`` layers where ``fc1.out_features``
    equals ``fc2.in_features``. Typical container names such as ``mlp``,
    ``ffn``, and ``feed_forward`` are inspected without relying exclusively on
    hardcoded attribute names.
    """

    mlp_pairs: List[Tuple[nn.Linear, nn.Linear]] = []
    for block in _iter_transformer_blocks(model):
        mlp_candidates = []
        for attr in ("mlp", "ffn", "feed_forward"):
            candidate = getattr(block, attr, None)
            if candidate is not None:
                mlp_candidates.append(candidate)

        for mlp in mlp_candidates:
            pairs = _linear_pairs_from_mlp(mlp)
            mlp_pairs.extend(pairs)

    return mlp_pairs


def _remove_pruning_linear(module: nn.Module) -> None:
    if hasattr(module, "weight_orig"):
        prune.remove(module, "weight")
    if hasattr(module, "bias_orig"):
        prune.remove(module, "bias")


def clear_pruning_from_mlp_pairs(mlp_pairs: Iterable[Tuple[nn.Linear, nn.Linear]]) -> None:
    """Remove pruning reparametrizations from the collected MLPs."""

    for fc1, fc2 in mlp_pairs:
        _remove_pruning_linear(fc1)
        _remove_pruning_linear(fc2)


def prune_mlp_hidden_units(
    mlp_pairs: Sequence[Tuple[nn.Linear, nn.Linear]], sparsity: float
) -> Tuple[int, int]:
    """Globally prune MLP hidden units using the ViT scoring method."""

    if sparsity <= 0:
        total_units = sum(fc1.weight.size(0) for fc1, _ in mlp_pairs)
        return total_units, 0

    units: List[Tuple[float, nn.Linear, nn.Linear, int]] = []
    for fc1, fc2 in mlp_pairs:
        weight1 = fc1.weight.detach()
        weight2 = fc2.weight.detach()
        hidden_dim = weight1.size(0)
        if hidden_dim != weight2.size(1):
            raise ValueError("fc1.out_features and fc2.in_features must match")

        bias = fc1.bias.detach() if fc1.bias is not None else None
        for k in range(hidden_dim):
            score = weight1[k].abs().mean().item() + weight2[:, k].abs().mean().item()
            if bias is not None:
                score += bias[k].abs().item()
            units.append((score, fc1, fc2, k))

    total_units = len(units)
    prune_units = int(math.floor(sparsity * total_units))
    if prune_units <= 0:
        return total_units, 0

    units.sort(key=lambda item: item[0])
    to_prune = units[:prune_units]

    rows_fc1: defaultdict[nn.Linear, set[int]] = defaultdict(set)
    cols_fc2: defaultdict[nn.Linear, set[int]] = defaultdict(set)
    for _, fc1, fc2, k in to_prune:
        rows_fc1[fc1].add(k)
        cols_fc2[fc2].add(k)

    for fc1, ks in rows_fc1.items():
        mask = torch.ones_like(fc1.weight)
        for k in ks:
            mask[k, :] = 0
        prune.custom_from_mask(fc1, name="weight", mask=mask)

        if fc1.bias is not None:
            bias_mask = torch.ones_like(fc1.bias)
            for k in ks:
                bias_mask[k] = 0
            prune.custom_from_mask(fc1, name="bias", mask=bias_mask)

    for fc2, ks in cols_fc2.items():
        mask = torch.ones_like(fc2.weight)
        for k in ks:
            mask[:, k] = 0
        prune.custom_from_mask(fc2, name="weight", mask=mask)

    return total_units, prune_units


@dataclass
class TransformerMLPPruningContext:
    """Small helper to encapsulate Transformer MLP pruning state."""

    enabled: bool
    mlp_pairs: Sequence[Tuple[nn.Linear, nn.Linear]]

    def clear_pruning(self) -> None:
        if not self.enabled:
            return
        clear_pruning_from_mlp_pairs(self.mlp_pairs)

    def prune(self, sparsity: float) -> Tuple[int, int]:
        if not self.enabled:
            return 0, 0
        return prune_mlp_hidden_units(self.mlp_pairs, sparsity)


def init_transformer_mlp_pruning(
    model: nn.Module, model_name: str
) -> TransformerMLPPruningContext:
    """Prepare a pruning context for ViT-like models without touching callers."""

    enabled = model_name in {"vit", "vit_b"}
    pairs = collect_transformer_mlp_pairs(model) if enabled else []
    return TransformerMLPPruningContext(enabled=enabled, mlp_pairs=pairs)
