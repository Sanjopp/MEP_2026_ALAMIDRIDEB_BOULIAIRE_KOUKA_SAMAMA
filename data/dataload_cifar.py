"""
Module de chargement du dataset CIFAR-10.
Utilise data_io pour abstraire la source (S3 ou local).
"""

import logging

import torch
from torch.utils.data import DataLoader, Dataset

from config.settings import (
    BATCH_SIZE,
    LOCAL_PATHS,
    NUM_WORKERS,
    S3_BUCKET,
    S3_KEYS,
)
from data.data_io import load_artifact

logger = logging.getLogger(__name__)


class CIFAR10UInt8Dataset(Dataset):
    """
    Dataset CIFAR-10 stocké en uint8 pour économiser la RAM.
    Normalisation float32 appliquée à la volée dans __getitem__.
    """

    def __init__(
        self,
        X_uint8: torch.Tensor,
        y: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        self.X_uint8 = X_uint8
        self.y = y
        self.mean = mean.float().view(3, 1, 1)
        self.std = std.float().view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X_uint8[idx].float().div(255.0).view(3, 32, 32)
        x = (x - self.mean) / self.std
        return x, self.y[idx]


def load_cifar10(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    force_local: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Retourne les DataLoaders train et test pour CIFAR-10.

    Args:
        batch_size:   Taille de batch.
        num_workers:  Nombre de workers DataLoader.
        force_local:  Bypass S3, charge depuis LOCAL_PATHS.

    Returns:
        Tuple (train_loader, test_loader).
    """
    artifact = load_artifact(
        bucket=S3_BUCKET,
        s3_key=S3_KEYS["cifar10"],
        local_path=LOCAL_PATHS["cifar10"],
        force_local=force_local,
    )

    train_ds = CIFAR10UInt8Dataset(
        artifact["X_train_uint8"],
        artifact["y_train"],
        artifact["mean"],
        artifact["std"],
    )
    test_ds = CIFAR10UInt8Dataset(
        artifact["X_test_uint8"],
        artifact["y_test"],
        artifact["mean"],
        artifact["std"],
    )

    pin_memory = torch.cuda.is_available()
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return (
        DataLoader(train_ds, shuffle=True,  **loader_kwargs),
        DataLoader(test_ds,  shuffle=False, **loader_kwargs),
    )