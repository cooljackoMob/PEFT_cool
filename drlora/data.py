"""Datasets for the two downstream tasks.

Both loaders are fixed across experimental groups (same splits, same
augmentation, same seed) -- only the rank allocation varies.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def classification_loaders(root: str, dataset: str = "cifar100", img_size: int = 224,
                           batch_size: int = 64, workers: int = 8, seed: int = 0,
                           subset: int | None = None, val_subset: int | None = None
                           ) -> Tuple[DataLoader, DataLoader, int]:
    """Classification loaders.

    `subset`/`val_subset` cap the number of examples (fixed seed), for cheap
    calibration runs; leave them None for the real experiment.
    """
    from torchvision import datasets, transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    if dataset == "hf_cifar100":
        # Same data as `cifar100`, fetched from the Hugging Face hub instead of
        # www.cs.toronto.edu -- useful where that host is unreachable.  Accepts
        # `subset`/`val_subset` so a calibration run can use a fixed slice.
        from datasets import load_dataset

        class _HF(Dataset):
            def __init__(self, split: str, tf, n: int | None):
                self.ds = load_dataset("uoft-cs/cifar100", split=split)
                if n:
                    self.ds = self.ds.shuffle(seed=seed).select(range(min(n, len(self.ds))))
                self.tf = tf

            def __len__(self):
                return len(self.ds)

            def __getitem__(self, i):
                r = self.ds[i]
                return self.tf(r["img"].convert("RGB")), int(r["fine_label"])

        tr = _HF("train", train_tf, subset)
        va = _HF("test", eval_tf, val_subset)
        n_cls = 100
    elif dataset == "cifar100":
        tr = datasets.CIFAR100(root, train=True, download=True, transform=train_tf)
        va = datasets.CIFAR100(root, train=False, download=True, transform=eval_tf)
        n_cls = 100
    elif dataset == "flowers102":
        tr = datasets.Flowers102(root, split="train", download=True, transform=train_tf)
        va = datasets.Flowers102(root, split="test", download=True, transform=eval_tf)
        n_cls = 102
    else:
        tr = datasets.ImageFolder(f"{root}/train", transform=train_tf)
        va = datasets.ImageFolder(f"{root}/val", transform=eval_tf)
        n_cls = len(tr.classes)

    g = torch.Generator().manual_seed(seed)
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=workers,
                   pin_memory=True, drop_last=True, generator=g),
        DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=workers,
                   pin_memory=True),
        n_cls,
    )


def segmentation_loaders(root: str, dataset: str = "voc", img_size: int = 224,
                         batch_size: int = 16, workers: int = 8, seed: int = 0
                         ) -> Tuple[DataLoader, DataLoader, int]:
    from torchvision import datasets, transforms
    from torchvision.transforms import InterpolationMode

    img_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    mask_tf = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST),
        transforms.PILToTensor(),
    ])

    class _Seg(Dataset):
        def __init__(self, base):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            img, mask = self.base[i]
            return img_tf(img), mask_tf(mask).squeeze(0).long()

    if dataset != "voc":
        raise KeyError("only Pascal VOC is wired up; add your dense dataset here")
    tr = _Seg(datasets.VOCSegmentation(root, year="2012", image_set="train",
                                       download=True))
    va = _Seg(datasets.VOCSegmentation(root, year="2012", image_set="val",
                                       download=True))
    g = torch.Generator().manual_seed(seed)
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=workers,
                   pin_memory=True, drop_last=True, generator=g),
        DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=workers,
                   pin_memory=True),
        21,
    )
