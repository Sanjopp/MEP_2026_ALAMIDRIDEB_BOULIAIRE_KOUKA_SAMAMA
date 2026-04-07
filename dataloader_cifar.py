from io import BytesIO
import boto3
import torch
from torch.utils.data import Dataset, DataLoader


class CIFAR10UInt8Dataset(Dataset):
    def __init__(self, X_uint8, y, mean, std):
        """
        X_uint8: torch.uint8 de shape (N, 3072)
        y: torch.long de shape (N,)
        mean: torch tensor de shape (3,)
        std: torch tensor de shape (3,)
        """
        self.X_uint8 = X_uint8
        self.y = y
        self.mean = mean.float().view(3, 1, 1)
        self.std = std.float().view(3, 1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X_uint8[idx].float().div(255.0).view(3, 32, 32)
        x = (x - self.mean) / self.std
        y = self.y[idx]
        return x, y


s3 = boto3.client("s3")


def load_cifar10_from_s3(
    bucket="data-mep-project",
    key="DATA/Cifar_10/cifar10_uint8.pt",
    batch_size=128,
    num_workers=0
):
    obj = s3.get_object(Bucket=bucket, Key=key)

    artifact = torch.load(
        BytesIO(obj["Body"].read()),
        map_location="cpu",
        weights_only=True
    )

    X_train_uint8 = artifact["X_train_uint8"]
    y_train = artifact["y_train"]
    X_test_uint8 = artifact["X_test_uint8"]
    y_test = artifact["y_test"]
    mean = artifact["mean"]
    std = artifact["std"]

    train_ds = CIFAR10UInt8Dataset(X_train_uint8, y_train, mean, std)
    test_ds = CIFAR10UInt8Dataset(X_test_uint8, y_test, mean, std)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader