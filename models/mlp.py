import torch.nn as nn
from .batchnorm import BatchNorm1D


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims=(512, 256, 128),
        num_classes: int = 10,
        use_bn: bool = False,
    ):
        """
        Generic MLP with optional BatchNorm.

        Parameters
        ----------
        input_dim : int
            Input dimension (e.g. 784 for Fashion-MNIST, 3072 for CIFAR-10)
        hidden_dims : tuple
            Sizes of hidden layers (default: 512, 256, 128)
        num_classes : int
            Number of output classes
        use_bn : bool
            Whether to use Batch Normalization
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_bn:
                layers.append(BatchNorm1D(h))
            layers.append(nn.ReLU())
            prev_dim = h

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
