"""ResNet18 variant without skip connections."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision.models.resnet import (  # type: ignore[attr-defined]
    ResNet,
    ResNet18_Weights,
    conv3x3,
)


class BasicBlockNoSkip(nn.Module):
    """Basic residual block without the residual connection."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1:
            raise ValueError("BasicBlockNoSkip only supports groups=1")
        if base_width != 64:
            raise ValueError("BasicBlockNoSkip only supports base_width=64")
        if dilation > 1:
            raise NotImplementedError("BasicBlockNoSkip does not support dilation > 1")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        # Keep attributes for API-compatibility even if they are unused in forward
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class ResNetNoSkip(ResNet):
    """ResNet backbone that swaps residual blocks with non-residual ones."""

    def __init__(self, block: type[BasicBlockNoSkip], layers: list[int], **kwargs) -> None:
        super().__init__(block, layers, **kwargs)


def resnet18_no_skip(
    *,
    weights: ResNet18_Weights | None = None,
    progress: bool = True,
    **kwargs: Any,
) -> ResNetNoSkip:
    """Builds a ResNet18 without skip connections.

    Parameters
    ----------
    weights: torchvision.models.ResNet18_Weights | None
        Pretrained weights are not supported for the no-skip variant. Pass ``None``.
    progress: bool
        Unused, kept for API compatibility.
    kwargs: Any
        Additional keyword arguments forwarded to :class:`ResNet`.
    """
    if weights is not None and not isinstance(weights, ResNet18_Weights):
        raise ValueError("weights must be a ResNet18_Weights enum or None")
    if isinstance(weights, ResNet18_Weights) and weights != ResNet18_Weights.DEFAULT:
        raise ValueError("Pretrained weights are not available for resnet18_no_skip.")
    if weights is not None:
        # We only accept the enum for compatibility, but we don't load the weights.
        kwargs.setdefault("num_classes", 1000)
    model = ResNetNoSkip(BasicBlockNoSkip, [2, 2, 2, 2], **kwargs)
    return model


__all__ = ["BasicBlockNoSkip", "ResNetNoSkip", "resnet18_no_skip"]
