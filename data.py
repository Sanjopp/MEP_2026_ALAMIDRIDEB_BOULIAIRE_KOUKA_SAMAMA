from io import BytesIO
import boto3
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

s3 = boto3.client("s3")


def unpickle_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pickle.load(BytesIO(obj["Body"].read()), encoding="bytes")


def load_cifar10_from_s3(bucket="data-mep-project", prefix="DATA/cifar-10-batches-py", batch_size=128):
    train_batches = [
        unpickle_s3(bucket, f"{prefix}/data_batch_{i}") for i in range(1, 6)
    ]

    X_train = np.concatenate([batch[b"data"] for batch in train_batches], axis=0)
    y_train = np.concatenate([batch[b"labels"] for batch in train_batches], axis=0)

    test_batch = unpickle_s3(bucket, f"{prefix}/test_batch")
    X_test = test_batch[b"data"]
    y_test = np.array(test_batch[b"labels"])

    X_train = preprocess_cifar(X_train)
    X_test = preprocess_cifar(X_test)

    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def preprocess_cifar(X):
    """
    X: (N, 3072) uint8
    return: torch.FloatTensor (N, 3, 32, 32)
    """
    X = X.astype(np.float32) / 255.0
    X = X.reshape(-1, 3, 32, 32)

    # Per-channel normalization (standard CIFAR-10)
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)

    X = (X - mean) / std
    return torch.from_numpy(X).float()