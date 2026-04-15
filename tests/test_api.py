"""Tests for the FastAPI application."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest

MOCK_ENV = {
    "MODEL_TYPE": "cnn",
    "DATASET": "cifar10",
    "USE_BN": "true",
    "S3_BUCKET": "data-mep-project",
    "S3_MODEL_KEY": "models/cifar10_cnn_bn.pt",
}


@pytest.fixture()
def mock_predictor():
    """Return a fully mocked Predictor instance."""
    predictor = MagicMock()
    predictor.model_type = "cnn"
    predictor.dataset = "cifar10"
    predictor.use_bn = True
    predictor.device = "cpu"
    predictor.predict.return_value = {
        "predicted_class": 3,
        "predicted_label": "cat",
        "confidence": 0.921,
        "probabilities": [0.01] * 9 + [0.921 - 0.09],
    }
    return predictor


@pytest.fixture()
def client(mock_predictor):
    """
    Return a TestClient with the Predictor mocked out.
    The lifespan is bypassed — we inject the mock directly.
    """
    with patch("api.main.predictor", mock_predictor):
        from api.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        """Health endpoint should return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        """Health response should match the HealthResponse schema."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_type"] == "cnn"
        assert data["dataset"] == "cifar10"
        assert data["use_bn"] is True
        assert data["device"] == "cpu"

    def test_health_returns_503_when_model_not_loaded(self):
        """Health endpoint should return 503 if predictor is None."""
        with patch("api.main.predictor", None):
            from api.main import app
            with TestClient(app, raise_server_exceptions=False) as c:
                response = c.get("/health")
        assert response.status_code == 503


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def _cifar10_payload(self) -> dict:
        """Return a valid CIFAR-10 image payload."""
        return {"image": [[[0.5] * 32] * 32] * 3}

    def test_predict_returns_200(self, client):
        """Predict endpoint should return HTTP 200 on valid input."""
        response = client.post("/predict", json=self._cifar10_payload())
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        """Predict response should match the PredictResponse schema."""
        response = client.post("/predict", json=self._cifar10_payload())
        data = response.json()
        assert "predicted_class" in data
        assert "predicted_label" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert isinstance(data["predicted_class"], int)
        assert isinstance(data["predicted_label"], str)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["probabilities"], list)
        assert len(data["probabilities"]) == 10

    def test_predict_correct_values(self, client):
        """Predict response should return the mocked prediction values."""
        response = client.post("/predict", json=self._cifar10_payload())
        data = response.json()
        assert data["predicted_class"] == 3
        assert data["predicted_label"] == "cat"
        assert data["confidence"] == 0.921

    def test_predict_returns_422_on_wrong_shape(self, client, mock_predictor):
        """Predict endpoint should return 422 if image shape is wrong."""
        mock_predictor.predict.side_effect = ValueError("Invalid image shape.")
        wrong_payload = {"image": [[[0.5] * 28] * 28] * 1}
        response = client.post("/predict", json=wrong_payload)
        assert response.status_code == 422

    def test_predict_returns_503_when_model_not_loaded(self):
        """Predict endpoint should return 503 if predictor is None."""
        with patch("api.main.predictor", None):
            from api.main import app
            with TestClient(app, raise_server_exceptions=False) as c:
                response = c.post("/predict", json={"image": [[[0.5] * 32] * 32] * 3})
        assert response.status_code == 503

    def test_predict_returns_500_on_inference_error(self, client, mock_predictor):
        """Predict endpoint should return 500 on unexpected inference error."""
        mock_predictor.predict.side_effect = RuntimeError("CUDA out of memory.")
        response = client.post("/predict", json=self._cifar10_payload())
        assert response.status_code == 500
