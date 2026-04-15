"""Tests for the FastAPI application."""

import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("MODEL_TYPE", "cnn")
os.environ.setdefault("DATASET", "cifar10")
os.environ.setdefault("USE_BN", "true")
os.environ.setdefault("S3_BUCKET", "data-mep-project")
os.environ.setdefault("S3_MODEL_KEY", "models/cifar10_cnn_bn.pt")

mock_s3_client = MagicMock()
mock_s3_client.download_file.return_value = None

with patch("boto3.client", return_value=mock_s3_client), patch(
    "torch.load", return_value={}
):
    from fastapi.testclient import TestClient

    import api.main as main_module
    from api.main import app


def make_mock_predictor():
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
def mock_predictor():
    return make_mock_predictor()


@pytest.fixture()
def client(mock_predictor):
    main_module.predictor = mock_predictor
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    main_module.predictor = None


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_type"] == "cnn"
        assert data["dataset"] == "cifar10"
        assert data["use_bn"] is True
        assert data["device"] == "cpu"

    def test_health_returns_503_when_model_not_loaded(self):
        """Health endpoint should return 503 if predictor is None."""
        main_module.predictor = None
        with patch("api.main.Predictor", side_effect=Exception("no s3")):
            with TestClient(app, raise_server_exceptions=False) as c:
                response = c.get("/health")
        assert response.status_code == 503


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def _cifar10_payload(self) -> dict:
        return {"image": [[[0.5] * 32] * 32] * 3}

    def test_predict_returns_200(self, client):
        response = client.post("/predict", json=self._cifar10_payload())
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
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
        response = client.post("/predict", json=self._cifar10_payload())
        data = response.json()
        assert data["predicted_class"] == 3
        assert data["predicted_label"] == "cat"
        assert data["confidence"] == 0.921

    def test_predict_returns_422_on_wrong_shape(self, client, mock_predictor):
        mock_predictor.predict.side_effect = ValueError("Invalid image shape.")
        wrong_payload = {"image": [[[0.5] * 28] * 28] * 1}
        response = client.post("/predict", json=wrong_payload)
        assert response.status_code == 422

    def test_predict_returns_503_when_model_not_loaded(self):
        """Predict endpoint should return 503 if predictor is None."""
        main_module.predictor = None
        with patch("api.main.Predictor", side_effect=Exception("no s3")):
            with TestClient(app, raise_server_exceptions=False) as c:
                response = c.post("/predict", json={"image": [[[0.5] * 32] * 32] * 3})
        assert response.status_code == 503

    def test_predict_returns_500_on_inference_error(self, client, mock_predictor):
        mock_predictor.predict.side_effect = RuntimeError("CUDA out of memory.")
        response = client.post("/predict", json=self._cifar10_payload())
        assert response.status_code == 500
