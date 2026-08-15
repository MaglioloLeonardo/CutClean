"""Utilities for loading the unbiased Waterbirds dataset."""

import os
from glob import glob
from typing import Callable, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data_setup import default_root

DEFAULT_WATERBIRDS_ROOT = os.path.join(default_root(), "waterbirds_unbiased")


class UnbiasedWaterbirdsFolders(Dataset):
    """Waterbirds images organized as ``root/{split}/{label}/*.png``.

    Filenames are expected to follow the pattern ``..._lbl<Y>_bias<B>.png`` to
    recover both the target label and the bias attribute.
    """

    def __init__(
        self, root: str, split: str, transform: Optional[Callable] = None
    ) -> None:
        assert split in {"train", "val", "test"}
        self.paths: Sequence[str] = sorted(glob(os.path.join(root, split, "*", "*.png")))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        path = self.paths[idx]
        name = os.path.basename(path)
        parts = name.split("_")
        label = int(parts[-2].replace("lbl", ""))
        bias = int(parts[-1].replace("bias", "").replace(".png", ""))

        with Image.open(path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # Return the class label as the task target and the bias as the private label.
        # The index is not needed by the training loop, which expects exactly
        # ``(image, target, private_label)``.
        return img, label, bias


DEFAULT_TRANSFORMS = transforms.Compose([transforms.ToTensor()])


def make_unbiased_waterbirds_dataloader(
    root: str = DEFAULT_WATERBIRDS_ROOT,
    split: str = "train",
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle: bool = False,
    transform: Optional[Callable] = DEFAULT_TRANSFORMS,
) -> DataLoader:
    """Create a :class:`DataLoader` for the unbiased Waterbirds dataset.

    The function keeps the original image resolution by default; no resizing is
    applied unless provided through ``transform``.
    """

    dataset = UnbiasedWaterbirdsFolders(root=root, split=split, transform=transform)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
