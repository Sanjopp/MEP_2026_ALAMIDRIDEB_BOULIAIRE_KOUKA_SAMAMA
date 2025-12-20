import torch
from torchvision import datasets, transforms


def get_fashion_mnist_loaders(batch_size=128):
    """
    Télécharge automatiquement Fashion-MNIST si absent,
    puis retourne train_loader et test_loader.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_ds = datasets.FashionMNIST(
        root="/home/kouka/Projectss/Advanced_ML_BN_2025/DATA",
        train=True,
        download=True,
        transform=transform
    )

    test_ds = datasets.FashionMNIST(
        root="/home/kouka/Projectss/Advanced_ML_BN_2025/DATA",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader
