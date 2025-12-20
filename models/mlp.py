import torch.nn as nn
from .batchnorm import BatchNorm1D


class MLP(nn.Module):
    def __init__(self, use_bn=False):
        super().__init__()
        self.use_bn = use_bn

        self.fc1 = nn.Linear(3 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        if use_bn:
            self.bn1 = BatchNorm1D(256)
            self.bn2 = BatchNorm1D(128)

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
