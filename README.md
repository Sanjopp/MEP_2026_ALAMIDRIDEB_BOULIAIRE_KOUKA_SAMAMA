# How Does Batch Normalization Really Help Optimization?

![CI](https://github.com/Sanjopp/MEP_2026_ALAMIDRIDEB_BOULIAIRE_KOUKA_SAMAMA/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-red)
![License](https://img.shields.io/badge/license-MIT-green)

**Advanced Machine Learning Course Project** — Z. Alami Drideb, Z. Bouliaire, H. Kouka, R. Samama

This project investigates the core question: *how does Batch Normalization really help optimization?* We conduct controlled experiments using MLP and CNN architectures trained with and without Batch Normalization on CIFAR-10 and Fashion-MNIST. The central hypothesis is that BN helps primarily through **loss landscape smoothing**, not by stabilizing activation statistics.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Research Design](#research-design)
3. [Project Structure](#project-structure)
4. [Models](#models)
5. [Training](#training)
6. [Dashboard & Experiment Tracking](#dashboard--experiment-tracking)
7. [API](#api)
8. [Deployment](#deployment)
9. [CI/CD](#cicd)
10. [Datasets](#datasets)
11. [References](#references)

---

## Quick Start

```bash
# 1. Install dependencies (requires uv)
make requirements

# 2. Configure secrets
cp config/secrets.env.example config/secrets.env
# edit config/secrets.env with AWS credentials and model config

# 3. Launch MLflow tracking UI
make mlflow-ui         # http://localhost:5000

# 4. Launch the Streamlit dashboard
make streamlit         # http://localhost:8501

# 5. Launch the REST API (dev mode)
make run-api           # http://localhost:8000
```

---

## Research Design

**Experimental Matrix**: 2 Datasets × 2 Architectures × 2 Analysis Levels = **8 Core Experiments**

| # | Dataset | Architecture | Granularity |
|---|---------|--------------|-------------|
| 1 | Fashion-MNIST | MLP | Epoch-level |
| 2 | Fashion-MNIST | MLP | Step-level (per-batch) |
| 3 | Fashion-MNIST | CNN | Epoch-level |
| 4 | Fashion-MNIST | CNN | Step-level |
| 5 | CIFAR-10 | MLP | Epoch-level |
| 6 | CIFAR-10 | MLP | Step-level |
| 7 | CIFAR-10 | CNN | Epoch-level |
| 8 | CIFAR-10 | CNN | Step-level |

Each experiment trains two models — with and without BN — and compares:
- Loss and accuracy curves
- Gradient norms (L2) per step/epoch
- Activation distributions (CIFAR-10 specific analysis)

**Expected findings:**
1. Models with BN converge faster
2. Loss curves are smoother with BN
3. Gradient norms are more stable
4. Final accuracy is higher with BN

---

## Project Structure

```
.
├── models/                     # Neural network architectures
│   ├── mlp.py                  # MLP with optional BatchNorm
│   ├── cnn.py                  # ConvMLP with optional BatchNorm
│   └── batchnorm.py            # Custom BatchNorm1D from scratch
│
├── src/training/               # Training pipeline
│   ├── train.py                # Main entry point (MLflow logging)
│   ├── config_handler.py       # CLI + YAML config management
│   ├── checkpoint.py           # Save/load model checkpoints
│   ├── reproducibility.py      # Seed management, device utils
│   └── evaluate.py             # Test-set evaluation
│
├── api/                        # FastAPI service
│   ├── main.py                 # Endpoints: /health, /predict, /train
│   ├── predictor.py            # Model loading (S3) + inference
│   └── schemas.py              # Pydantic request/response schemas
│
├── app/
│   └── streamlit_app.py        # Interactive experiment dashboard
│
├── data/
│   ├── dataload_cifar.py       # CIFAR-10 loader (uint8, S3-backed)
│   ├── dataload_fashion.py     # Fashion-MNIST loader (IDX format)
│   └── data_io.py              # S3 / local fallback abstraction
│
├── config/
│   ├── settings.py             # Paths, S3 keys, DataLoader defaults
│   └── secrets.env.example     # Secrets template
│
├── tests/
│   ├── test_models.py          # Unit tests: MLP, CNN, BatchNorm
│   └── test_api.py             # Unit tests: API endpoints
│
├── helm/                       # Kubernetes Helm chart
├── argocd/                     # ArgoCD application manifests
├── config.yaml                 # Default training configuration
├── Dockerfile                  # Multi-stage image (Python 3.13-slim)
├── docker-compose.yml          # Local container setup
├── Makefile                    # All common commands
└── pyproject.toml              # Dependencies (uv)
```

---

## Models

### MLP (Multi-Layer Perceptron)

```python
MLP(
    input_dim=784,              # 784 for Fashion-MNIST, 3072 for CIFAR-10
    hidden_dims=(512, 256, 128),
    num_classes=10,
    use_bn=True                 # Toggle BatchNorm on/off
)
```

### ConvMLP (Convolutional + Dense)

```python
ConvMLP(
    in_channels=3,              # 3 for CIFAR-10, 1 for Fashion-MNIST
    input_size=32,              # 32 for CIFAR-10, 28 for Fashion-MNIST
    conv_channels=(32, 64),
    hidden_dims=(512, 256, 128),
    num_classes=10,
    use_bn=True
)
```

### Custom BatchNorm1D

`models/batchnorm.py` is a from-scratch implementation using PyTorch autograd:
- Computes batch mean and variance
- Normalizes activations (ε = 1e-5)
- Learnable scale (γ) and shift (β)
- Tracks running statistics for eval mode

---

## Training

### Via config file

Edit `config.yaml`, then:

```bash
python src/training/train.py
```

Key parameters in `config.yaml`:

```yaml
dataset:
  name: "cifar10"        # cifar10 | fashion_mnist
  batch_size: 128

model:
  type: "cnn"            # mlp | cnn
  use_bn: true

training:
  num_epochs: 10
  learning_rate: 0.1
  optimizer: "adam"      # adam | sgd
  seed: 42
```

### Via CLI

```bash
python src/training/train.py --dataset cifar10 --model_type cnn --use_bn true --num_epochs 20
```

### Grid search + best model

Run a full grid search across configurations and automatically upload the best model to S3:

```bash
make train-gridsearch
```

Results and metrics are logged to MLflow and the best checkpoint is pushed to S3 under `models/`.

---

## Dashboard & Experiment Tracking

### Streamlit dashboard

```bash
make streamlit
```

Opens at `http://localhost:8501`. Features:
- Browse all MLflow experiments
- Filter by dataset, model type, BatchNorm flag
- Plot learning curves (loss, accuracy, gradient norms)
- Launch new training runs from the UI
- Run inference directly from the browser

### MLflow UI

```bash
make mlflow-ui
```

Opens at `http://localhost:5000`. All training runs are automatically logged with:
- Hyperparameters (lr, optimizer, seed, use_bn, ...)
- Per-epoch metrics (loss, accuracy, gradient norm)
- Model artifacts

---

## API

The FastAPI service loads a trained model from S3 at startup and exposes two main endpoints.

### Environment variables

Copy `config/secrets.env.example` to `config/secrets.env` and fill in:

| Variable | Description | Example |
|----------|-------------|---------|
| `MODEL_TYPE` | Model architecture | `cnn` or `mlp` |
| `DATASET` | Dataset the model was trained on | `cifar10` or `fashion_mnist` |
| `USE_BN` | Whether BatchNorm is enabled | `true` or `false` |
| `S3_BUCKET` | S3 bucket name | `data-mep-project` |
| `S3_MODEL_KEY` | Path to model weights in S3 | `models/cifar10_cnn_bn.pt` |
| `AWS_ACCESS_KEY_ID` | AWS access key | — |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | — |
| `AWS_DEFAULT_REGION` | Bucket region | `eu-north-1` |

### Run locally

```bash
make requirements
make run-api
```

API available at `http://localhost:8000`.  
Swagger docs at `http://localhost:8000/docs`.

### Run with Docker

```bash
make docker-build
make docker-up          # or make docker-up-detached
make docker-logs
make docker-down
```

### Endpoints

**Health check:**

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_type": "cnn",
  "dataset": "cifar10",
  "use_bn": true,
  "device": "cpu"
}
```

**Prediction:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": [[[0.5, ...]]]}'
```

```json
{
  "predicted_class": 3,
  "predicted_label": "cat",
  "confidence": 0.921,
  "probabilities": [0.01, 0.01, 0.01, 0.921, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
}
```

**Launch training:**

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"dataset": "cifar10", "model_type": "cnn", "use_bn": true, "num_epochs": 10}'
```

### Tests

```bash
make test          # all tests
make test-api      # API only
make test-models   # models only
```

---

## Deployment

### SSP Cloud (Kubernetes / Helm)

**Prerequisites:**
- Access to [SSP Cloud](https://datalab.sspcloud.fr)
- `kubectl` and `helm` configured (see **My account > Kubernetes**)
- AWS secrets stored in SSP Cloud **My Secrets** under `mep-secrets`

```bash
make helm-deploy      # deploy
make helm-status      # check status
make helm-uninstall   # tear down
```

Live URL: `https://batchnorm-api.user-sanjopp.lab.sspcloud.fr`

To switch model, edit `helm/values.yaml`:

```yaml
env:
  MODEL_TYPE: mlp
  DATASET: fashion_mnist
  USE_BN: "false"
  S3_MODEL_KEY: models/fashion_mnist_mlp_no_bn.pt
```

Then redeploy with `make helm-deploy`.

### ArgoCD (GitOps)

ArgoCD manifests are in `argocd/`. The application syncs automatically from the `main` branch, watching `helm/` for changes.

```bash
kubectl apply -f argocd/application.yaml
```

---

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on every push and pull request:

| Job | Trigger | What it does |
|-----|---------|-------------|
| `lint` | push / PR | Ruff format check + linting |
| `test` | push / PR | pytest (models + API) |
| `docker` | push to `main` | Build & push image to Docker Hub |

The Docker job only runs if lint and test pass.

---

## Datasets

### CIFAR-10

- 32×32 RGB images, 10 classes
- 50,000 train + 10,000 test
- S3: `data-mep-project/DATA/Cifar_10/cifar10_uint8.pt` (~150 MB)

### Fashion-MNIST

- 28×28 grayscale images, 10 classes
- 60,000 train + 10,000 test
- S3: `data-mep-project/DATA/FashionMNIST/raw/` (~52 MB)

Data is loaded from S3 at training time with automatic fallback to `data/local/` if offline.

To sync data locally:

```bash
make sync_data_down    # S3 → local
make sync_data_up      # local → S3
```

---

## References

- [Batch Normalization: Accelerating Deep Network Training...](https://arxiv.org/abs/1502.03167) — Ioffe & Szegedy (2015)
- [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604) — Santurkar et al. (2018)
