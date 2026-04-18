"""Pydantic schemas for API request and response validation."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Input schema for prediction endpoint."""

    image: list[list[list[float]]] = Field(
        ...,
        description="Image as nested list of floats. Shape: [C, H, W]. "
        "CIFAR-10: [3, 32, 32], Fashion-MNIST: [1, 28, 28].",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "image": [[[0.5] * 32] * 32] * 3,
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    """Output schema for prediction endpoint."""

    predicted_class: int = Field(..., description="Predicted class index (0-9).")
    predicted_label: str = Field(..., description="Human-readable class label.")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0).")
    probabilities: list[float] = Field(
        ..., description="Softmax probabilities for all 10 classes."
    )


class HealthResponse(BaseModel):
    """Output schema for health check endpoint."""

    status: str = Field(..., description="Service status.")
    model_type: str = Field(..., description="Loaded model type (mlp or cnn).")
    dataset: str = Field(..., description="Dataset the model was trained on.")
    use_bn: bool = Field(..., description="Whether the model uses Batch Normalization.")
    device: str = Field(..., description="Device used for inference (cpu or cuda).")
