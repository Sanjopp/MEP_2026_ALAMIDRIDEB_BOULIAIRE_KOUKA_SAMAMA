"""
Module de chargement du dataset FashionMNIST.
Les fichiers sont stockés au format IDX binaire sur S3.
"""

import logging
from pathlib import Path
import struct

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config.settings import (
    BATCH_SIZE,
    LOCAL_PATHS,
    NUM_WORKERS,
    S3_BUCKET,
    S3_KEYS,
)
from data.data_io import load_idx_artifact

logger = logging.getLogger(__name__)


# ── Parseurs IDX ──────────────────────────────────────────────────────────────

def _parse_idx_images(data: bytes) -> torch.Tensor:
    """
    Parse un fichier IDX d'images.
    Format : [magic:4B][n:4B][rows:4B][cols:4B][pixels: n*rows*cols bytes]

    Returns:
        Tensor uint8 de shape (N, 784).
    """
    magic, n, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 2051:
        raise ValueError(f"Magic number invalide pour images IDX : {magic}")
    pixels = np.frombuffer(data[16:], dtype=np.uint8)
    return torch.from_numpy(pixels.reshape(n, rows * cols).copy())


def _parse_idx_labels(data: bytes) -> torch.Tensor:
    """
    Parse un fichier IDX de labels.
    Format : [magic:4B][n:4B][labels: n bytes]

    Returns:
        Tensor long de shape (N,).
    """
    magic,  = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"Magic number invalide pour labels IDX : {magic}")
    labels = np.frombuffer(data[8:], dtype=np.uint8)
    return torch.from_numpy(labels.astype(np.int64).copy())


# ── Calcul mean / std ─────────────────────────────────────────────────────────

def _compute_mean_std(x_uint8: torch.Tensor) -> tuple[float, float]:
    """
    Calcule mean et std depuis le train set uniquement.
    x_uint8 : (N, 784) uint8 → converti en float32 pour le calcul.

    Returns:
        Tuple (mean, std) en float.
    """
    x_float = x_uint8.float().div(255.0)
    mean = x_float.mean().item()
    std  = x_float.std().item()
    logger.info("Mean calculée : %.4f | Std calculée : %.4f", mean, std)
    return mean, std


# ── Dataset ───────────────────────────────────────────────────────────────────

class FashionMNISTUInt8Dataset(Dataset):
    """
    Dataset FashionMNIST stocké en uint8 pour économiser la RAM.
    Normalisation float32 appliquée à la volée dans __getitem__.
    Images : 1 canal, 28x28.
    """

    def __init__(
        self,
        x_uint8: torch.Tensor,
        y: torch.Tensor,
        mean: float,
        std: float,
    ) -> None:
        self.x_uint8 = x_uint8
        self.y = y
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.x_uint8[idx].float().div(255.0).view(1, 28, 28)
        x = (x - self.mean) / self.std
        return x, self.y[idx]


# ── Chargement ────────────────────────────────────────────────────────────────

def _load_split(
    images_key: str,
    labels_key: str,
    images_local: Path | None,
    labels_local: Path | None,
    force_local: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Charge et parse une split (train ou test)."""
    x = _parse_idx_images(load_idx_artifact(S3_BUCKET, images_key, images_local, force_local))
    y = _parse_idx_labels(load_idx_artifact(S3_BUCKET, labels_key, labels_local, force_local))
    return x, y


def load_fashion_mnist(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    val_split: float = 0.2,
    force_local: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Retourne les DataLoaders train, val et test pour FashionMNIST.

    Split: Train (80%) | Val (20%) | Test (10k séparé)

    Args:
        batch_size:   Taille de batch.
        num_workers:  Nombre de workers DataLoader.
        val_split:    Proportion du train set pour validation (défaut 0.2 = 20%).
        force_local:  Bypass S3, charge depuis LOCAL_PATHS.

    Returns:
        Tuple (train_loader, val_loader, test_loader).
    """
    x_train, y_train = _load_split(
        images_key   = S3_KEYS["fashion_train_images"],
        labels_key   = S3_KEYS["fashion_train_labels"],
        images_local = LOCAL_PATHS["fashion_train_images"],
        labels_local = LOCAL_PATHS["fashion_train_labels"],
        force_local  = force_local,
    )
    x_test, y_test = _load_split(
        images_key   = S3_KEYS["fashion_test_images"],
        labels_key   = S3_KEYS["fashion_test_labels"],
        images_local = LOCAL_PATHS["fashion_test_images"],
        labels_local = LOCAL_PATHS["fashion_test_labels"],
        force_local  = force_local,
    )

    # Calcul mean/std depuis le train set uniquement
    mean, std = _compute_mean_std(x_train)

    # Split train en train/val
    n_train = len(y_train)
    n_val = int(n_train * val_split)
    n_train_actual = n_train - n_val

    logger.info(f"FashionMNIST Split: Train={n_train_actual} | Val={n_val} | Test={len(y_test)}")

    # Créer les datasets
    train_ds = FashionMNISTUInt8Dataset(
        x_train[:n_train_actual],
        y_train[:n_train_actual],
        mean,
        std,
    )
    val_ds = FashionMNISTUInt8Dataset(
        x_train[n_train_actual:],
        y_train[n_train_actual:],
        mean,
        std,
    )
    test_ds = FashionMNISTUInt8Dataset(
        x_test,
        y_test,
        mean,
        std,
    )

    pin_memory = torch.cuda.is_available()
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return (
        DataLoader(train_ds, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, shuffle=False, **loader_kwargs),
    )