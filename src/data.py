import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.data_io import ensure_dataset_local


def unpickle(file: Path):
    with file.open("rb") as fo:
        data = pickle.load(fo, encoding="bytes")
    return data


def load_cifar10_from_batches():
    """
    Charge CIFAR-10 depuis le dossier local synchronisé depuis S3.
    Le dossier est obtenu via ensure_dataset_local("cifar10").
    """
    root = Path(ensure_dataset_local("cifar10"))

    expected_files = [
        "batches.meta",
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
        "test_batch",
    ]
    missing = [f for f in expected_files if not (root / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"CIFAR-10 files missing after S3 sync: {missing}. "
            "Check S3 prefix datasets/cifar-10-batches-py and config/config.yaml."
        )

    train_batches = [unpickle(root / f"data_batch_{i}") for i in range(1, 6)]

    X_train = np.concatenate([batch[b"data"] for batch in train_batches], axis=0)
    y_train = np.concatenate([batch[b"labels"] for batch in train_batches], axis=0)

    test_batch = unpickle(root / "test_batch")
    X_test = test_batch[b"data"]
    y_test = np.array(test_batch[b"labels"])

    return X_train, y_train, X_test, y_test


def preprocess_cifar(X: np.ndarray) -> torch.Tensor:
    """
    X: (N, 3072) uint8
    return: torch.FloatTensor (N, 3, 32, 32)
    """
    X = X.astype(np.float32) / 255.0
    X = X.reshape(-1, 3, 32, 32)

    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)

    X = (X - mean) / std
    return torch.from_numpy(X).float()


def get_dataloaders(batch_size: int = 128):
    """
    Retourne les DataLoaders CIFAR-10.
    Les données sont automatiquement synchronisées depuis S3.
    """
    X_train, y_train, X_test, y_test = load_cifar10_from_batches()

    X_train = preprocess_cifar(X_train)
    X_test = preprocess_cifar(X_test)

    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
