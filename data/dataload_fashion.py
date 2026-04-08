"""
Module de chargement du dataset FashionMNIST.
Les fichiers sont stockés au format IDX binaire sur S3.
"""

import logging
import struct
from pathlib import Path
from typing import Union

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
from data.data_io import load_bytes_from_s3

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
    magic, n = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"Magic number invalide pour labels IDX : {magic}")
    labels = np.frombuffer(data[8:], dtype=np.uint8)
    return torch.from_numpy(labels.astype(np.int64).copy())


# ── Calcul mean / std ─────────────────────────────────────────────────────────

def _compute_mean_std(X_uint8: torch.Tensor) -> tuple[float, float]:
    """
    Calcule mean et std depuis le train set uniquement.
    X_uint8 : (N, 784) uint8 → converti en float32 pour le calcul.

    Returns:
        Tuple (mean, std) en float.
    """
    X_float = X_uint8.float().div(255.0)
    mean = X_float.mean().item()
    std  = X_float.std().item()
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
        X_uint8: torch.Tensor,
        y: torch.Tensor,
        mean: float,
        std: float,
    ) -> None:
        self.X_uint8 = X_uint8
        self.y = y
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X_uint8[idx].float().div(255.0).view(1, 28, 28)
        x = (x - self.mean) / self.std
        return x, self.y[idx]


# ── Chargement ────────────────────────────────────────────────────────────────

def _get_bytes(
    s3_key: str,
    local_path: Union[Path, None],
    force_local: bool,
) -> bytes:
    """Récupère les bytes depuis S3 ou local selon le contexte."""
    if force_local or (local_path and Path(local_path).exists()):
        logger.info("Chargement local : %s", local_path)
        return Path(local_path).read_bytes()
    return load_bytes_from_s3(S3_BUCKET, s3_key)


def _load_split(
    images_key: str,
    labels_key: str,
    images_local: Union[Path, None],
    labels_local: Union[Path, None],
    force_local: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Charge et parse une split (train ou test)."""
    X = _parse_idx_images(_get_bytes(images_key, images_local, force_local))
    y = _parse_idx_labels(_get_bytes(labels_key, labels_local, force_local))
    return X, y


def load_fashion_mnist(
    batch_size: int  = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    force_local: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Retourne les DataLoaders train et test pour FashionMNIST.

    Args:
        batch_size:   Taille de batch.
        num_workers:  Nombre de workers DataLoader.
        force_local:  Bypass S3, charge depuis LOCAL_PATHS.

    Returns:
        Tuple (train_loader, test_loader).
    """
    X_train, y_train = _load_split(
        images_key   = S3_KEYS["fashion_train_images"],
        labels_key   = S3_KEYS["fashion_train_labels"],
        images_local = LOCAL_PATHS["fashion_train_images"],
        labels_local = LOCAL_PATHS["fashion_train_labels"],
        force_local  = force_local,
    )
    X_test, y_test = _load_split(
        images_key   = S3_KEYS["fashion_test_images"],
        labels_key   = S3_KEYS["fashion_test_labels"],
        images_local = LOCAL_PATHS["fashion_test_images"],
        labels_local = LOCAL_PATHS["fashion_test_labels"],
        force_local  = force_local,
    )

    # Calcul depuis le train set uniquement (jamais depuis le test !)
    mean, std = _compute_mean_std(X_train)

    pin_memory = torch.cuda.is_available()
    loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )

    return (
        DataLoader(FashionMNISTUInt8Dataset(X_train, y_train, mean, std), shuffle=True,  **loader_kwargs),
        DataLoader(FashionMNISTUInt8Dataset(X_test,  y_test,  mean, std), shuffle=False, **loader_kwargs),
    )