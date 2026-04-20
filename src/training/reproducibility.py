"""
Reproducibility utilities for ensuring deterministic behavior.
"""

import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seeds for all libraries to ensure reproducibility.

    Args:
        seed: Random seed value
    """
    # Python random
    import random

    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set environment variables
    os.environ["PYTHONHASHSEED"] = str(seed)

    # For full reproducibility (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed} for reproducibility")


def get_device() -> torch.device:
    """
    Get the appropriate device (GPU if available, else CPU).

    Returns:
        torch.device: The device to use for training
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


class MetricsTracker:
    """Track training and validation metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "grad_norm": [],
        }

    def update(
        self,
        train_loss: float | None = None,
        train_acc: float | None = None,
        val_loss: float | None = None,
        val_acc: float | None = None,
        grad_norm: float | None = None,
    ) -> None:
        """
        Update metrics.

        Args:
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            grad_norm: Gradient norm
        """
        if train_loss is not None:
            self.history["train_loss"].append(train_loss)
        if train_acc is not None:
            self.history["train_acc"].append(train_acc)
        if val_loss is not None:
            self.history["val_loss"].append(val_loss)
        if val_acc is not None:
            self.history["val_acc"].append(val_acc)
        if grad_norm is not None:
            self.history["grad_norm"].append(grad_norm)

    def get_dict(self) -> dict:
        """Return metrics as dictionary."""
        return self.history

    def __repr__(self) -> str:
        """String representation."""
        lines = ["Metrics History:"]
        for key, values in self.history.items():
            if values:
                lines.append(f"  {key}: {len(values)} entries, last={values[-1]:.6f}")
        return "\n".join(lines)
