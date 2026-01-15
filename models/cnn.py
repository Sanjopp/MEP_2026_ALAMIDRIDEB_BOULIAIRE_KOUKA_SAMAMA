import torch.nn as nn
from .batchnorm import BatchNorm1D


class ConvMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_size: int,
        conv_channels=(32, 64),
        hidden_dims=(512, 256, 128),
        num_classes: int = 10,
        use_bn: bool = False,
    ):
        """
        Simple CNN with optional BatchNorm.

        Parameters
        ----------
        in_channels : int
            3 pour CIFAR-10, 1 pour Fashion-MNIST
        input_size : int
            32 (CIFAR) ou 28 (Fashion)
        conv_channels : tuple
            Nombre de canaux pour chaque conv layer
        hidden_dims : tuple
            Sizes of hidden layers (default: 512, 256, 128)
        num_classes : int
            Number of output classes
        use_bn : bool
            Whether to use Batch Normalization
        """
        super().__init__()

        layers = []
        prev_c = in_channels
        size = input_size

        # convolutions
        for c in conv_channels:
            layers.append(nn.Conv2d(prev_c, c, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            size //= 2
            prev_c = c

        self.conv = nn.Sequential(*layers)
        flat_dim = prev_c * size * size
        mlp_layers = []
        prev_dim = flat_dim

        for h in hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, h))
            if use_bn:
                mlp_layers.append(BatchNorm1D(h))
            mlp_layers.append(nn.ReLU())
            prev_dim = h

        mlp_layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.mlp(x)
