import os
from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


def get_transforms(image_size: int = 224, is_train: bool = True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def get_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Returns train, val, test dataloaders and class names.
    Expects data_dir to contain one sub-folder per Pokemon class.
    """
    full_dataset = datasets.ImageFolder(data_dir, transform=get_transforms(image_size, is_train=False))
    class_names = full_dataset.classes
    n = len(full_dataset)

    generator = torch.Generator().manual_seed(seed)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test], generator=generator)

    # Apply augmentation only to training split via a wrapper
    train_ds.dataset = datasets.ImageFolder(data_dir, transform=get_transforms(image_size, is_train=True))
    # val and test keep the original (non-augmented) dataset reference
    val_test_ds = datasets.ImageFolder(data_dir, transform=get_transforms(image_size, is_train=False))
    val_ds.dataset = val_test_ds
    test_ds.dataset = val_test_ds

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names
