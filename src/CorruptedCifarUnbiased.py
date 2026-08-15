import os
from typing import Optional, Tuple, List

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class UnbiasedCorruptedCIFAR10(Dataset):
    """
    Dataset for the UNBIASED CIFAR10-C produced by the generation script:

        unbiased_cifar10c_{percent}_joint_splits/
            train/
                {class}/img_xxxxxx_lbl{class}_bias{bias}.png
            valid/
                ...
            test/
                ...

    Each sample is returned as:
        (image_tensor, class_label, bias_label)
    where bias_label plays the role of the privacy attribute.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        extensions=(".png", ".jpg", ".jpeg"),
        nb_classes: int = 10,
    ):
        """
        Args:
            root: UNBIASED root folder, e.g.
                  /path/unbiased_cifar10c_5pct_joint_splits
            split: 'train', 'val', 'valid' or 'test'
            transform: torchvision transforms to apply
            extensions: image extensions to consider
        """
        self.root = root
        self.transform = transform
        self.extensions = tuple(e.lower() for e in extensions)

        # On disk the folder is named "valid", while the training code often
        # says "val": accept both.
        if split == "val":
            split_dir_name = "valid"
        else:
            split_dir_name = split

        self.split = split_dir_name
        self.split_dir = os.path.join(root, split_dir_name)

        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Directory dello split non trovata: {self.split_dir}")

        # Collect all image paths by walking the per-class folders 0,1,...
        self.image_paths: List[str] = []
        labels: List[int] = []
        biases: List[int] = []
        skipped_samples = 0
        for class_name in sorted(os.listdir(self.split_dir)):
            class_dir = os.path.join(self.split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(self.extensions):
                    path = os.path.join(class_dir, fname)
                    label, bias = self._parse_class_and_bias_from_path(path)
                    if 0 <= label < nb_classes:
                        self.image_paths.append(path)
                        labels.append(label)
                        biases.append(bias)
                    else:
                        skipped_samples += 1

        if skipped_samples:
            print(
                f"[UnbiasedCorruptedCIFAR10] Skipped {skipped_samples} samples "
                f"with out-of-range labels for {nb_classes} classes in split '{split}'."
            )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"Nessuna immagine trovata in {self.split_dir}")

        # Expose target and private attribute in the form expected by the training pipeline
        self.targets = torch.tensor(labels, dtype=torch.long)
        self.bias_targets = torch.tensor(biases, dtype=torch.long)
        self.target = "targets"
        self.bias_attr = "bias_targets"

    def __len__(self) -> int:
        return len(self.image_paths)

    def _parse_class_and_bias_from_path(self, path: str) -> Tuple[int, int]:
        """
        Extract class and bias from the path, following exactly the naming
        pattern used when the dataset was written:

            .../{class}/img_000123_lbl{class}_bias{bias}.png
        """
        class_dir = os.path.basename(os.path.dirname(path))
        try:
            class_from_dir = int(class_dir)
        except ValueError:
            raise RuntimeError(f"Impossibile interpretare '{class_dir}' come classe (dirname). Path: {path}")

        fname = os.path.basename(path)
        # Example: "img_000123_lbl3_bias7.png"
        parts = fname.split("_")
        lbl_part = None
        bias_part = None
        for p in parts:
            if p.startswith("lbl"):
                lbl_part = p
            if p.startswith("bias"):
                bias_part = p

        if lbl_part is None or bias_part is None:
            raise RuntimeError(
                f"Nome file non conforme al pattern '..._lblX_biasY.png': {fname}"
            )

        try:
            class_from_name = int(lbl_part.replace("lbl", ""))
        except ValueError:
            raise RuntimeError(f"Impossibile estrarre la classe da '{lbl_part}' in {fname}")

        bias_str = os.path.splitext(bias_part)[0].replace("bias", "")
        try:
            bias = int(bias_str)
        except ValueError:
            raise RuntimeError(f"Impossibile estrarre il bias da '{bias_part}' in {fname}")

        if class_from_dir != class_from_name:
            raise RuntimeError(
                f"Incoerenza tra classe da directory ({class_from_dir}) "
                f"e da nome file ({class_from_name}) per path: {path}"
            )

        return class_from_dir, bias

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]

        with Image.open(path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        label, bias = self._parse_class_and_bias_from_path(path)

        # Return (x, y, s)
        return img, label, bias


def make_dataloader(
    root: str,
    split: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    image_size: int = 32,
    # kept for backward compatibility with the previous API: ignored
    metadata_csv: Optional[str] = None,
    target_col: Optional[str] = None,
    private_col: Optional[str] = None,
    split_col: Optional[str] = None,
    image_dir: Optional[str] = None,
) -> DataLoader:
    """
    Build a DataLoader for the UNBIASED CIFAR10-C.

    IMPORTANT:
    - reads no CSV
    - uses no metadata
    - metadata_csv, target_col, private_col, split_col and image_dir are ignored
      (they exist only to stay compatible with the existing calling code)

    Args:
        root: base folder, e.g.
              /path/unbiased_cifar10c_5pct_joint_splits
        split: 'train', 'val'/'valid', 'test'
        batch_size: batch size
        num_workers: DataLoader num_workers
        shuffle: whether to shuffle the indices
        image_size: images are resized to (image_size, image_size)
    """
    def _compute_mean_std(resize_first: transforms.Compose) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (root, image_size)
        if not hasattr(_compute_mean_std, "_cache"):
            _compute_mean_std._cache = {}

        if cache_key in _compute_mean_std._cache:
            return _compute_mean_std._cache[cache_key]

        stats_split = "train" if os.path.isdir(os.path.join(root, "train")) else split
        stats_dataset = UnbiasedCorruptedCIFAR10(
            root=root,
            split=stats_split,
            transform=resize_first,
        )

        stats_loader = DataLoader(
            stats_dataset,
            batch_size=min(512, batch_size),
            shuffle=False,
            num_workers=max(1, num_workers),
            pin_memory=True,
        )

        mean = torch.zeros(3)
        var = torch.zeros(3)
        total_images = 0

        for images, _, _ in stats_loader:
            images = images.view(images.size(0), images.size(1), -1)
            batch_mean = images.mean(dim=2)
            batch_var = images.var(dim=2, unbiased=False)

            mean += batch_mean.sum(dim=0)
            var += batch_var.sum(dim=0)
            total_images += images.size(0)

        mean /= total_images
        std = torch.sqrt(var / total_images)
        _compute_mean_std._cache[cache_key] = (mean, std)
        return mean, std

    resize_and_tensor = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
    )

    mean, std = _compute_mean_std(resize_and_tensor)

    if split == "train":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            # optional augmentation
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
        ])

    dataset = UnbiasedCorruptedCIFAR10(
        root=root,
        split=split,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader


if __name__ == "__main__":
    # Quick manual smoke test
    root = "/path/assoluto/a/unbiased_cifar10c_5pct_joint_splits"

    dl = make_dataloader(
        root=root,
        split="train",
        batch_size=8,
        num_workers=0,
        shuffle=False,
        image_size=224,
    )

    for imgs, cls, bias in dl:
        print("Batch immagini:", imgs.shape)
        print("Classi:", cls[:8].tolist())
        print("Bias:  ", bias[:8].tolist())
        break
