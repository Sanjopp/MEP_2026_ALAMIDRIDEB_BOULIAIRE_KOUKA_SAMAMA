"""FastAPI application exposing model inference."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException

from api.predictor import Predictor
from api.schemas import HealthResponse, PredictRequest, PredictResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global predictor
    logger.info("Starting up — loading model from S3...")
    try:
        if predictor is None:
            predictor = Predictor()
        logger.info("Model loaded successfully. API is ready.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    logger.info("Shutting down.")
    predictor = None


app = FastAPI(
    title="Batch Normalization Classifier API",
    description=(
        "REST API for image classification using MLP and CNN models "
        "trained with and without Batch Normalization on CIFAR-10 and Fashion-MNIST."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the service status and loaded model information.",
)
def health() -> HealthResponse:
    """Health check endpoint."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return HealthResponse(
        status="ok",
        model_type=predictor.model_type,
        dataset=predictor.dataset,
        use_bn=predictor.use_bn,
        device=str(predictor.device),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict image class",
    description=(
        "Accepts a single image as a nested list of floats [C, H, W] "
        "and returns the predicted class, label, confidence score, "
        "and full probability distribution."
    ),
)
def predict(request: PredictRequest) -> PredictResponse:
    """Prediction endpoint."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        result = predictor.predict(request.image)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed.") from e
    return PredictResponse(**result)

from fastapi import FastAPI, HTTPException
from api.schemas import TrainingRequest
import subprocess
from pathlib import Path

app = FastAPI()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@app.post("/train")
def launch_training(request: TrainingRequest):
    cmd = [
        "uv", "run", "python", "src/training/train.py",
        "--dataset", request.dataset,
        "--model-type", request.model_type,
        "--use-bn", str(request.use_bn),
        "--batch-size", str(request.batch_size),
        "--learning-rate", str(request.learning_rate),
        "--num-epochs", str(request.num_epochs),
        "--optimizer", request.optimizer,
        "--seed", str(request.seed),
    ]

    if request.verbose:
        cmd.append("--verbose")

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur lors du lancement du training: {exc}")

    return {
        "status": "success" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "command": " ".join(cmd),
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }