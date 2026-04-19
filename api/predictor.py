"""Model loading and inference logic."""

import logging
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv
import torch
import torch.nn.functional as func

from models.cnn import ConvMLP
from models.mlp import MLP

load_dotenv()

logger = logging.getLogger(__name__)

CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

FASHION_MNIST_LABELS = [
    "t-shirt/top",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
]

DATASET_LABELS = {
    "cifar10": CIFAR10_LABELS,
    "fashion_mnist": FASHION_MNIST_LABELS,
}

DATASET_INPUT_SPECS = {
    "cifar10": {"input_dim": 3072, "in_channels": 3, "input_size": 32},
    "fashion_mnist": {"input_dim": 784, "in_channels": 1, "input_size": 28},
}


class Predictor:
    """Handles model loading from S3 and inference."""

    def __init__(self) -> None:
        self.model_type = os.environ["MODEL_TYPE"].lower()
        self.dataset = os.environ["DATASET"].lower()
        self.use_bn = os.environ["USE_BN"].lower() == "true"
        self.s3_bucket = os.environ["S3_BUCKET"]
        self.s3_model_key = os.environ["S3_MODEL_KEY"]

        if self.dataset not in DATASET_LABELS:
            raise ValueError(
                f"Unknown dataset '{self.dataset}'. Expected one of: {list(DATASET_LABELS.keys())}"
            )
        if self.model_type not in ("mlp", "cnn"):
            raise ValueError(f"Unknown model type '{self.model_type}'. Expected 'mlp' or 'cnn'.")

        self.labels = DATASET_LABELS[self.dataset]
        self.input_specs = DATASET_INPUT_SPECS[self.dataset]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model_from_s3()

        logger.info(
            f"Predictor ready — model={self.model_type}, dataset={self.dataset}, "
            f"use_bn={self.use_bn}, device={self.device}"
        )

    def _build_model(self) -> torch.nn.Module:
        """Instantiate the model architecture."""
        if self.model_type == "mlp":
            return MLP(
                input_dim=self.input_specs["input_dim"],
                hidden_dims=(512, 256, 128),
                num_classes=10,
                use_bn=self.use_bn,
            )
        return ConvMLP(
            in_channels=self.input_specs["in_channels"],
            input_size=self.input_specs["input_size"],
            conv_channels=(32, 64),
            hidden_dims=(512, 256, 128),
            num_classes=10,
            use_bn=self.use_bn,
        )

    def _load_model_from_s3(self) -> torch.nn.Module:
        """Download model weights from S3 and load into architecture."""
        local_path = Path("/tmp/model.pt")

        logger.info(f"Downloading model from s3://{self.s3_bucket}/{self.s3_model_key} ...")

        s3 = boto3.client("s3")
        s3.download_file(self.s3_bucket, self.s3_model_key, str(local_path))

        logger.info("Model downloaded. Loading weights...")

        model = self._build_model()
        state_dict = torch.load(local_path, map_location=self.device)

        # Load state dict with strict=False to ignore missing running_mean/running_var
        model.load_state_dict(state_dict, strict=False)

        model.to(self.device)
        model.eval()

        logger.info("Model loaded and ready for inference.")
        return model

    @torch.no_grad()
    def predict(self, image: list[list[list[float]]]) -> dict:
        """
        Run inference on a single image.

        Args:
            image: Nested list of floats with shape [C, H, W].

        Returns:
            Dictionary with predicted_class, predicted_label,
            confidence, and probabilities.
        """
        tensor = torch.tensor(image, dtype=torch.float32)

        expected_shape = (
            self.input_specs["in_channels"],
            self.input_specs["input_size"],
            self.input_specs["input_size"],
        )
        if tensor.shape != torch.Size(expected_shape):
            raise ValueError(
                f"Invalid image shape {tuple(tensor.shape)}. "
                f"Expected {expected_shape} for dataset '{self.dataset}'."
            )

        # Images are already normalized by the DataLoader
        tensor = tensor.unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probabilities = func.softmax(logits, dim=1).squeeze(0)

        predicted_class = int(probabilities.argmax().item())
        confidence = float(probabilities[predicted_class].item())
        probs_list = probabilities.tolist()

        return {
            "predicted_class": predicted_class,
            "predicted_label": self.labels[predicted_class],
            "confidence": round(confidence, 6),
            "probabilities": [round(p, 6) for p in probs_list],
        }
