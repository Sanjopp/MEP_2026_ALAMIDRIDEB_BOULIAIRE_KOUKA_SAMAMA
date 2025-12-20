import torch.nn as nn
from .batchnorm import BatchNorm1D


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims=(256, 128),
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
            Sizes of hidden layers
        num_classes : int
            Number of output classes
        use_bn : bool
            Whether to use Batch Normalization
        """
        super().__init__()
        self.use_bn = use_bn

        h1, h2 = hidden_dims

        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, num_classes)

        if use_bn:
            self.bn1 = BatchNorm1D(h1)
            self.bn2 = BatchNorm1D(h2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.fc2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.relu(x)

        return self.fc3(x)
