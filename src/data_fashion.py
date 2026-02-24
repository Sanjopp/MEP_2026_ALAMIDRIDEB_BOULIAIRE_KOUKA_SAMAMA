import torch
from torchvision import datasets, transforms

from src.data_io import ensure_dataset_local


def get_fashion_mnist_loaders(batch_size: int = 128):
    """
    Retourne les DataLoaders Fashion-MNIST.
    Les données sont synchronisées depuis S3 via ensure_dataset_local("fashion_mnist").
    Aucun téléchargement externe n'est autorisé.
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )

    root = str(ensure_dataset_local("fashion_mnist"))

    train_ds = datasets.FashionMNIST(
        root=root,
        train=True,
        download=False,
        transform=transform,
    )

    test_ds = datasets.FashionMNIST(
        root=root,
        train=False,
        download=False,
        transform=transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader
