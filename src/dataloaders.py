
import os
import time
import random

import numpy as np
import torch

from data_setup import default_root
from datasets.celebA import CelebA
from CorruptedCifarUnbiased import (
    UnbiasedCorruptedCIFAR10,
    make_dataloader as make_unbiased_cifar10c_dataloader,
)
from waterbirds_dataloader import (
    DEFAULT_WATERBIRDS_ROOT,
    make_unbiased_waterbirds_dataloader,
)


def _seed_worker_fn(base_seed):
    def _seed_worker(worker_id):
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _seed_worker


def _build_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _resolve_unbiased_cifar_root(args, percent: str):
    """Return the first existing root for the unbiased corrupted CIFAR-10-C dataset."""

    preferred_root = getattr(
        args, "corruptedcifarunbiased_root", os.path.join(default_root(), "corrupted_cifar_unbiased")
    )
    candidates = [preferred_root]

    # If the preferred root does not directly contain the splits, try the historical subfolder name.
    candidates.append(os.path.join(preferred_root, f"unbiased_cifar10c_{percent}_joint_splits"))

    # Fallback to the generic datapath logic as a last resort.
    candidates.append(os.path.join(args.datapath, f"unbiased_cifar10c_{percent}_joint_splits"))

    for root in candidates:
        if os.path.isdir(root):
            return root

    # If nothing exists, return the preferred root so that downstream errors are informative.
    return preferred_root


def _resolve_unbiased_waterbirds_root(args):
    """Return the first existing root for the unbiased Waterbirds dataset."""

    preferred_root = getattr(args, "waterbirds_root", DEFAULT_WATERBIRDS_ROOT)
    candidates = [preferred_root, os.path.join(args.datapath, "waterbirds_unbiased")]

    for root in candidates:
        if os.path.isdir(root):
            return root

    return preferred_root

def build_dataloaders(
    args,
    dataset_attributes=None,
    num_workers=[8, 4, 4],
):
    dataset_attributes = dataset_attributes or {}
    dataset = args.dataset.split("-")[0]
    dataset_parts = args.dataset.split("-")
    worker_seed_fn = _seed_worker_fn(args.seed)
    if dataset == "celeba":
        target = args.dataset.split("-")[1]
        bias_attr = args.dataset.split("-")[2]
        unbiased = False if len(args.dataset.split("-")) > 3 and args.dataset.split("-")[3] == "biased" else True
        print(f"target:{target}")
        print(f"bias_attr:{bias_attr}")
        print(f"unbiased:{unbiased}")
        train_dataset = CelebA(
            args.datapath + dataset + "/",
            split="train",
            target=target,
            bias_attr=bias_attr,
            unbiased=unbiased,
            seed=args.seed,
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers[0],
            pin_memory=True,
            worker_init_fn=worker_seed_fn,
            generator=_build_generator(args.seed),
        )

        val_dataset = CelebA(
            args.datapath + dataset + "/",
            split="valid",
            target=target,
            bias_attr=bias_attr,
            unbiased=unbiased,
            seed=args.seed,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers[1],
            pin_memory=True,
            worker_init_fn=worker_seed_fn,
            generator=_build_generator(args.seed + 1),
        )

        test_dataset = CelebA(
            args.datapath + dataset + "/",
            split="test",
            target=target,
            bias_attr=bias_attr,
            unbiased=unbiased,
            seed=args.seed,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers[2],
            pin_memory=True,
            worker_init_fn=worker_seed_fn,
            generator=_build_generator(args.seed + 2),
        )
    elif dataset == "corruptedcifarunbiased":
        percent = dataset_parts[1] if len(dataset_parts) > 1 else dataset_attributes.get("percent", "5pct")
        dataset_root = _resolve_unbiased_cifar_root(args, percent)
        high_res_models = {"resnet18", "resnet18_noskip", "vit", "vit_b", "tiny_vit"}
        target_image_size = 224 if args.model in high_res_models else 32
        train_loader = make_unbiased_cifar10c_dataloader(
            root=dataset_root,
            split="train",
            batch_size=args.batch_size,
            num_workers=num_workers[0],
            shuffle=True,
            image_size=target_image_size,
        )
        val_loader = make_unbiased_cifar10c_dataloader(
            root=dataset_root,
            split="val",
            batch_size=args.batch_size,
            num_workers=num_workers[1],
            shuffle=True,
            image_size=target_image_size,
        )
        test_loader = make_unbiased_cifar10c_dataloader(
            root=dataset_root,
            split="test",
            batch_size=args.batch_size,
            num_workers=num_workers[2],
            shuffle=True,
            image_size=target_image_size,
        )
    elif dataset == "unbiasedWaterbirds":
        dataset_root = _resolve_unbiased_waterbirds_root(args)
        train_loader = make_unbiased_waterbirds_dataloader(
            root=dataset_root,
            split="train",
            batch_size=args.batch_size,
            num_workers=num_workers[0],
            shuffle=True,
        )
        val_loader = make_unbiased_waterbirds_dataloader(
            root=dataset_root,
            split="val",
            batch_size=args.batch_size,
            num_workers=num_workers[1],
            shuffle=True,
        )
        test_loader = make_unbiased_waterbirds_dataloader(
            root=dataset_root,
            split="test",
            batch_size=args.batch_size,
            num_workers=num_workers[2],
            shuffle=True,
        )
    return train_loader, val_loader, test_loader


from torch.amp import autocast

def build_dataloaders_for_ph(args, model, ph, dls):
    dl_list = {i: {"train": [], "val": [], "test": []} for i in args.used_phs}
    for split in ["train", "val", "test"]:
        print(f"Building dl for split {split}")
        dl2 = torch.utils.data.DataLoader(
            dataset=dls[split].dataset,
            batch_size=(max(args.batch_size//2, 16)),
            shuffle=False,  # Keep sequential for efficient processing
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive between iterations
            prefetch_factor=2,  # Prefetch next 2 batches while processing current one
            worker_init_fn=_seed_worker_fn(args.seed + 3),
            generator=_build_generator(args.seed + 3),
        )
        bottleneck_activations = []
        private_labels = []

       

        t = time.time()
        with torch.no_grad(), autocast("cuda"):
            print("Loading data in the model")
            for batch, (data, _, priv_labels) in enumerate(dl2):
                data = data.to(args.device)
                _ = model(data)
                bottleneck_activations.append(ph.bottleneck.output.detach().to("cpu"))
                private_labels.append(priv_labels)
        t2 = time.time()
        print(f"Time to load data in the model: {t2-t}")

        private_labels = torch.cat(private_labels)

        
        t3 = time.time()
        print(f"ph - Concatenating activations")
        bottleneck_activations = torch.cat(bottleneck_activations)
        print(bottleneck_activations.shape, private_labels.shape)
        dl_list[split] = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(
                bottleneck_activations, private_labels
            ),
            batch_size=args.batch_size,
            shuffle=True,  # Shuffle only in the final DataLoader
            num_workers=8,
            pin_memory=True,
            worker_init_fn=_seed_worker_fn(args.seed + 4),
            generator=_build_generator(args.seed + 4),
        )
        t4 = time.time()
        print(f"---- Time to concatenate: {t4-t3}")

    return dl_list
