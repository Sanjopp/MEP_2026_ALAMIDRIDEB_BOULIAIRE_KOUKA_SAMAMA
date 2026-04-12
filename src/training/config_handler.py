"""
Configuration management module.
Handles loading from YAML files and CLI argument overrides.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

def str_to_bool(value):
    """Convert string to boolean."""
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', '1', 'yes', 'y'}:
        return True
    elif value.lower() in {'false', '0', 'no', 'n'}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


class ConfigManager:
    """Manages configuration from YAML files and CLI arguments."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_yaml()

    def load_yaml(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file {self.config_path} not found. Using defaults.")
            self.config = self._get_defaults()
            return

        with open(self.config_path, "r") as f:
            loaded = yaml.safe_load(f)
            if loaded is None:
                self.config = self._get_defaults()
            else:
                self.config = loaded
        logger.info(f"Loaded configuration from {self.config_path}")

    @staticmethod
    def _get_defaults() -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "dataset": {
                "name": "cifar10",
                "batch_size": 128,
                "num_workers": 0,
                "val_split": 0.2,
            },
            "model": {
                "type": "cnn",
                "use_bn": True,
                "mlp": {
                    "input_dim": 3072,
                    "hidden_dims": [512, 256, 128],
                    "num_classes": 10,
                },
                "cnn": {
                    "in_channels": 3,
                    "input_size": 32,
                    "conv_channels": [32, 64],
                    "hidden_dims": [512, 256, 128],
                    "num_classes": 10,
                },
            },
            "training": {
                "num_epochs": 50,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "seed": 42,
            },
            "logging": {
                "log_interval": 10,
                "save_dir": "./results",
                "save_model": True,
                "save_metrics": True,
                "save_checkpoint": True,
                "verbose": True,
            },
        }

    def update_from_cli(self, args: argparse.Namespace) -> None:
        """
        Update configuration from CLI arguments.

        Args:
            args: Parsed command-line arguments
        """
        cli_updates = {k: v for k, v in vars(args).items() if v is not None}

        # Update dataset config
        if "dataset" in cli_updates:
            self.config["dataset"]["name"] = cli_updates["dataset"]
        if "batch_size" in cli_updates:
            self.config["dataset"]["batch_size"] = cli_updates["batch_size"]
        if "val_split" in cli_updates:
            self.config["dataset"]["val_split"] = cli_updates["val_split"]
        # Update model config
        if "model_type" in cli_updates:
            self.config["model"]["type"] = cli_updates["model_type"]
        if "use_bn" in cli_updates:
            self.config["model"]["use_bn"] = cli_updates["use_bn"]

        # Update training config
        if "learning_rate" in cli_updates:
            self.config["training"]["learning_rate"] = cli_updates["learning_rate"]
        if "num_epochs" in cli_updates:
            self.config["training"]["num_epochs"] = cli_updates["num_epochs"]
        if "optimizer" in cli_updates:
            self.config["training"]["optimizer"] = cli_updates["optimizer"]
        if "seed" in cli_updates:
            self.config["training"]["seed"] = cli_updates["seed"]

        # Update logging config
        if "save_dir" in cli_updates:
            self.config["logging"]["save_dir"] = cli_updates["save_dir"]
        if "verbose" in cli_updates:
            self.config["logging"]["verbose"] = cli_updates["verbose"]

        logger.info("Updated configuration from CLI arguments")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., "training.learning_rate")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.config[key]

    def __repr__(self) -> str:
        """String representation of configuration."""
        return yaml.dump(self.config, default_flow_style=False)
    
def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train neural network with optional Batch Normalization"
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "fashion_mnist"],
        help="Dataset to train on (cifar10 or fashion_mnist)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for training",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        help="Validation split ratio",
    )

    # Model arguments
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["mlp", "cnn"],
        help="Model architecture (mlp or cnn)",
    )
    parser.add_argument(
        "--use-bn",
        type=str_to_bool,
        default=None,
        help="Use Batch Normalization (True or False)",
    )

    # Training arguments
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "sgd"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )

    # Logging arguments
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Directory to save results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser
