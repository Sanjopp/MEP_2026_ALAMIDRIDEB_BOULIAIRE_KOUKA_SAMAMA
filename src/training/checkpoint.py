"""
Model and checkpoint management utilities.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


class ModelCheckpoint:
    """Manage model checkpoints and metadata."""

    def __init__(self, save_dir: str = "./results"):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint directory: {self.save_dir}")

    def save_model(
        self,
        model: torch.nn.Module,
        filename: str = "model.pt",
    ) -> Path:
        """
        Save model state dict.

        Args:
            model: PyTorch model
            filename: Filename for the model

        Returns:
            Path to saved model
        """
        path = self.save_dir / filename
        torch.save(model.state_dict(), path)
        logger.info(f"Model saved to {path}")
        return path

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, Any],
        filename: str = "checkpoint.pt",
    ) -> Path:
        """
        Save training checkpoint with model, optimizer, and metadata.

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch number
            metrics: Dictionary of metrics
            filename: Filename for the checkpoint

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }
        path = self.save_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
        return path

    def load_model(
        self,
        model: torch.nn.Module,
        filename: str = "model.pt",
    ) -> torch.nn.Module:
        """
        Load model state dict.

        Args:
            model: PyTorch model to load into
            filename: Filename of the saved model

        Returns:
            Model with loaded state
        """
        path = self.save_dir / filename
        if not path.exists():
            logger.warning(f"Model file {path} not found")
            return model

        model.load_state_dict(torch.load(path, map_location="cpu"))
        logger.info(f"Model loaded from {path}")
        return model

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        filename: str = "checkpoint.pt",
    ) -> Dict[str, Any]:
        """
        Load training checkpoint.

        Args:
            model: PyTorch model to load into
            optimizer: PyTorch optimizer to load into
            filename: Filename of the checkpoint

        Returns:
            Dictionary with checkpoint data (epoch, metrics)
        """
        path = self.save_dir / filename
        if not path.exists():
            logger.warning(f"Checkpoint file {path} not found")
            return {"epoch": 0, "metrics": {}}

        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Checkpoint loaded from {path}")

        return {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
        }

    def save_metrics(
        self,
        metrics: Dict[str, Any],
        filename: str = "metrics.json",
    ) -> Path:
        """
        Save metrics as JSON.

        Args:
            metrics: Dictionary of metrics
            filename: Filename for the metrics

        Returns:
            Path to saved metrics file
        """
        path = self.save_dir / filename

        # Convert lists to ensure JSON serialization
        json_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, list):
                json_metrics[key] = value
            else:
                json_metrics[key] = value

        with open(path, "w") as f:
            json.dump(json_metrics, f, indent=2)

        logger.info(f"Metrics saved to {path}")
        return path

    def save_config(
        self,
        config: Dict[str, Any],
        filename: str = "config.json",
    ) -> Path:
        """
        Save configuration as JSON.

        Args:
            config: Configuration dictionary
            filename: Filename for the config

        Returns:
            Path to saved config file
        """
        path = self.save_dir / filename
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Configuration saved to {path}")
        return path

    def save_summary(
        self,
        summary: Dict[str, Any],
        filename: str = "summary.json",
    ) -> Path:
        """
        Save training summary.

        Args:
            summary: Summary dictionary
            filename: Filename for the summary

        Returns:
            Path to saved summary file
        """
        path = self.save_dir / filename
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to {path}")
        return path
