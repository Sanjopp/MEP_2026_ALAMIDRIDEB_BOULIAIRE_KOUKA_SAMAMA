import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class MiniVGG(nn.Module):
    """
    Mini VGG architecture on 3 different blocks
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3x32x32 -> 64x16x16
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 64x16x16 -> 128x8x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 128x8x8 -> 256x4x4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_cifar10_loaders(batch_size=64):
    transform = T.Compose([T.ToTensor()])
    # train dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    # test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    # dataloeaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def train(
    model, train_loader, test_loader, device, epochs=1, lr=3e-4, weight_decay=1e-4
):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = CosineAnnealingLR(optim, T_max=epochs)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        sched.step()

        train_loss = running_loss / total
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, test_loader, device)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "minivgg_cifar10_best.pt")

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"Test loss {test_loss:.4f} acc {test_acc:.3f} | "
            f"Best {best_acc:.3f}"
        )

    print("Saved best model to minivgg_cifar10_best.pt")


device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader, test_loader = get_cifar10_loaders(batch_size=128)

model = MiniVGG(num_classes=10)
train(model, train_loader, test_loader, device, epochs=1, lr=3e-4)
