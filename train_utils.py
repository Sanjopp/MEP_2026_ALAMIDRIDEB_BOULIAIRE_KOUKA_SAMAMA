import torch
import torch.nn.functional as F
from tqdm import tqdm


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5

def train_steps(model, loader, optimizer, device, history):
    """
    Entraîne le modèle sur UN epoch
    mais log les métriques à CHAQUE step.
    """
    model.train()

    for x, y in tqdm(loader, desc="Training steps", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        gnorm = grad_norm(model)
        optimizer.step()

        # Calcul de l'accuracy sur le batch
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()

        # LOG PAR STEP
        history["loss"].append(loss.item())
        history["grad_norm"].append(gnorm)
        history["train_acc"].append(acc)

def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum, grad_sum = 0.0, 0.0

    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        gnorm = grad_norm(model)
        optimizer.step()

        loss_sum += loss.item()
        grad_sum += gnorm

    return loss_sum / len(loader), grad_sum / len(loader)



@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total
