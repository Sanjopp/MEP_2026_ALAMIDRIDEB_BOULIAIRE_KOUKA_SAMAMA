import os
from pathlib import Path

# ── Racine du projet ──────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── S3 ───────────────────────────────────────────────────────────────────────
S3_BUCKET = os.getenv("S3_BUCKET", "data-mep-project")
VAL_SPLIT = float(os.getenv("VAL_SPLIT", 0.2))
S3_KEYS = {
    "cifar10":              "DATA/Cifar_10/cifar10_uint8.pt",
    "fashion_train_images": "DATA/FashionMNIST/raw/train-images-idx3-ubyte",
    "fashion_train_labels": "DATA/FashionMNIST/raw/train-labels-idx1-ubyte",
    "fashion_test_images":  "DATA/FashionMNIST/raw/t10k-images-idx3-ubyte",
    "fashion_test_labels":  "DATA/FashionMNIST/raw/t10k-labels-idx1-ubyte",
}

# ── Local fallback ────────────────────────────────────────────────────────────
LOCAL_DATA_DIR = Path(os.getenv("LOCAL_DATA_DIR", ROOT_DIR / "data" / "local"))

LOCAL_PATHS = {
    "cifar10":              LOCAL_DATA_DIR / "cifar10" / "cifar10_uint8.pt",
    "fashion_train_images": LOCAL_DATA_DIR / "fashion" / "train-images-idx3-ubyte",
    "fashion_train_labels": LOCAL_DATA_DIR / "fashion" / "train-labels-idx1-ubyte",
    "fashion_test_images":  LOCAL_DATA_DIR / "fashion" / "t10k-images-idx3-ubyte",
    "fashion_test_labels":  LOCAL_DATA_DIR / "fashion" / "t10k-labels-idx1-ubyte",
}

# ── DataLoader ────────────────────────────────────────────────────────────────
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", 128))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 0))