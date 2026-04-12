"""
Main training script for reproducible model training.

Usage:
    python train.py --config config.yaml
    python train.py --dataset cifar10 --model-type cnn --use-bn True --num-epochs 50
"""

from datetime import datetime
import logging
from pathlib import Path
import sys

from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.dataload_cifar import load_cifar10
from data.dataload_fashion import load_fashion_mnist
from models.cnn import ConvMLP
from models.mlp import MLP
from src.training.checkpoint import ModelCheckpoint
from src.training.config_handler import ConfigManager, create_argument_parser
from src.training.reproducibility import MetricsTracker, get_device, set_seed

load_dotenv()  # Load environment variables from .env file


def setup_logging(verbose: bool = True) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_dataset(config: ConfigManager, device: torch.device):
    """Load the specified dataset."""
    dataset_name = config.get("dataset.name")
    batch_size = config.get("dataset.batch_size")
    num_workers = config.get("dataset.num_workers")
    val_split = config.get("dataset.val_split", 0.2)

    logging.info(f"Loading {dataset_name} dataset...")

    if dataset_name == "cifar10":
        train_loader, val_loader, test_loader = load_cifar10(
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
        )
        input_dim = 3072
        in_channels = 3
        input_size = 32
    elif dataset_name == "fashion_mnist":
        train_loader, val_loader, test_loader = load_fashion_mnist(
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
        )
        input_dim = 784
        in_channels = 1
        input_size = 28
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    logging.info(
        f"Dataset loaded. Batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}"
    )

    return train_loader, val_loader, test_loader, input_dim, in_channels, input_size


def build_model(
    config: ConfigManager, input_dim: int, in_channels: int, input_size: int, device: torch.device
) -> nn.Module:
    """Build the model based on configuration."""
    model_type = config.get("model.type")
    use_bn = config.get("model.use_bn")
    num_classes = config.get(f"model.{model_type}.num_classes")

    logging.info(f"Building {model_type.upper()} model with BatchNorm={use_bn}...")

    if model_type == "mlp":
        hidden_dims = config.get("model.mlp.hidden_dims")
        model = MLP(
            input_dim=input_dim,
            hidden_dims=tuple(hidden_dims),
            num_classes=num_classes,
            use_bn=use_bn,
        )
    elif model_type == "cnn":
        conv_channels = config.get("model.cnn.conv_channels")
        hidden_dims = config.get("model.cnn.hidden_dims")
        model = ConvMLP(
            in_channels=in_channels,
            input_size=input_size,
            conv_channels=tuple(conv_channels),
            hidden_dims=tuple(hidden_dims),
            num_classes=num_classes,
            use_bn=use_bn,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model created with {total_params:,} parameters")

    return model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int = 10,
) -> tuple:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    grad_norms = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Compute gradient norm
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm**0.5
        grad_norms.append(grad_norm)

        optimizer.step()

        # Metrics
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        total_correct += pred.eq(target).sum().item()
        total_samples += data.size(0)

        if batch_idx % log_interval == 0:
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
            pbar.set_postfix(
                {"loss": f"{avg_loss:.4f}", "acc": f"{accuracy:.4f}", "gnorm": f"{grad_norm:.4f}"}
            )

    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0

    return avg_loss, avg_accuracy, avg_grad_norm


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]")

    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        total_correct += pred.eq(target).sum().item()
        total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{accuracy:.4f}"})

    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy


@torch.no_grad()
def evaluate_final(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Évaluation finale sur le TEST SET.
    Ce set n'a JAMAIS été utilisé pendant l'entraînement.

    Returns:
        Tuple (test_loss, test_accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(test_loader, desc="Evaluating Test Set")

    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        total_correct += pred.eq(target).sum().item()
        total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{accuracy:.4f}"})

    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy


def train(config: ConfigManager) -> None:
    """Main training function."""
    logger = logging.getLogger(__name__)

    # Set seed for reproducibility
    seed = config.get("training.seed")
    set_seed(seed)

    # Device
    device = get_device()

    # Load dataset
    train_loader, val_loader, test_loader, input_dim, in_channels, input_size = load_dataset(
        config, device
    )

    # Build model
    model = build_model(config, input_dim, in_channels, input_size, device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_type = config.get("training.optimizer")
    learning_rate = config.get("training.learning_rate")

    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    logger.info(f"Using {optimizer_type.upper()} optimizer with lr={learning_rate}")

    # Training loop
    num_epochs = config.get("training.num_epochs")
    log_interval = config.get("logging.log_interval")
    metrics_tracker = MetricsTracker()

    logger.info(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc, grad_norm = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, log_interval
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)

        # Track metrics
        metrics_tracker.update(
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            grad_norm=grad_norm,
        )

        logger.info(
            f"Epoch {epoch + 1:3d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Grad Norm: {grad_norm:.4f}"
        )

    logger.info("Training completed!")
    return model, metrics_tracker, test_loader


def save_results(
    model: nn.Module,
    metrics_tracker: MetricsTracker,
    config: ConfigManager,
    test_loss: float | None = None,
    test_acc: float | None = None,
) -> None:
    """Save model, metrics, and configuration."""
    save_dir = config.get("logging.save_dir")
    checkpoint_mgr = ModelCheckpoint(save_dir)

    # Create experiment identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = config.get("model.type")
    use_bn = config.get("model.use_bn")
    dataset = config.get("dataset.name")
    experiment_id = f"{dataset}_{model_type}_bn{use_bn}_{timestamp}"

    exp_dir = Path(save_dir) / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_mgr.save_dir = exp_dir

    # Save model
    if config.get("logging.save_model"):
        checkpoint_mgr.save_model(model, "model.pt")

    # Save metrics
    if config.get("logging.save_metrics"):
        checkpoint_mgr.save_metrics(metrics_tracker.get_dict(), "metrics.json")

    # Save config
    checkpoint_mgr.save_config(config.config, "config.json")

    # Get metrics history
    metrics_dict = metrics_tracker.get_dict()

    # Build comprehensive summary
    summary = {
        "dataset": dataset,
        "model_type": model_type,
        "use_bn": use_bn,
        "optimizer": config.get("training.optimizer"),
        "learning_rate": config.get("training.learning_rate"),
        "num_epochs": config.get("training.num_epochs"),
        "seed": config.get("training.seed"),
        # Final epoch metrics
        "final_train_loss": metrics_dict["train_loss"][-1] if metrics_dict["train_loss"] else None,
        "final_train_acc": metrics_dict["train_acc"][-1] if metrics_dict["train_acc"] else None,
        "final_val_loss": metrics_dict["val_loss"][-1] if metrics_dict["val_loss"] else None,
        "final_val_acc": metrics_dict["val_acc"][-1] if metrics_dict["val_acc"] else None,
        # Best metrics
        "best_train_acc": max(metrics_dict["train_acc"]) if metrics_dict["train_acc"] else 0,
        "best_val_acc": max(metrics_dict["val_acc"]) if metrics_dict["val_acc"] else 0,
        # Test metrics
        "test_loss": test_loss,
        "test_acc": test_acc,
        # Config for later analysis
        "config": config.config,
    }
    checkpoint_mgr.save_summary(summary, "summary.json")

    logging.info(f"Results saved to {exp_dir}")
    logging.info(f"Experiment ID: {experiment_id}")


def main():
    """Main entry point."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Load and update configuration
    config = ConfigManager(args.config)
    config.update_from_cli(args)

    # Setup logging
    setup_logging(config.get("logging.verbose"))
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(config)
    logger.info("=" * 80)

    # Train
    try:
        model, metrics_tracker, test_loader = train(config)

        # FINAL EVALUATION ON TEST SET
        device = get_device()
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate_final(model, test_loader, criterion, device)

        logger.info("=" * 80)
        logger.info("FINAL TEST SET EVALUATION")
        logger.info(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
        logger.info("=" * 80)

        # Save results
        if config.get("logging.save_model") or config.get("logging.save_metrics"):
            save_results(model, metrics_tracker, config, test_loss, test_acc)

        logger.info("=" * 80)
        logger.info("TRAINING SUCCESSFUL")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
