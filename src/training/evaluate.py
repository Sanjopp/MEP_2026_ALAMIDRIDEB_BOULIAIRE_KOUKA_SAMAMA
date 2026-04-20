"""
Full grid search over all model configs, then upload the best per config to S3.

Usage:
    python src/training/evaluate.py

Requires S3_BUCKET env var (or set in .env).
"""

import itertools
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
import torch.nn as nn

from src.training.config_handler import ConfigManager
from src.training.reproducibility import get_device
from src.training.train import evaluate_final, save_results, train

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

RESULTS_DIR = Path("./results")
S3_BUCKET = os.environ.get("S3_BUCKET", "data-mep-project")
S3_PREFIX = "models"
SEED = 42

ALL_CONFIGS = list(
    itertools.product(
        ["cifar10", "fashion_mnist"],
        ["mlp", "cnn"],
        [True, False],
    )
)

GRID_LR = [0.001, 0.01, 0.1]
GRID_BATCH_SIZES = [64, 128, 256]
GRID_EPOCHS = [10, 30]
GRID_OPTIMIZERS = ["adam", "sgd"]


# training


def _build_config(dataset, model_type, use_bn, lr, batch_size, num_epochs, optimizer):
    config = ConfigManager.__new__(ConfigManager)
    config.config_path = Path("config.yaml")
    config.config = {
        "dataset": {"name": dataset, "batch_size": batch_size, "num_workers": 0, "val_split": 0.2},
        "model": {
            "type": model_type,
            "use_bn": use_bn,
            "mlp": {"input_dim": 3072, "hidden_dims": [512, 256, 128], "num_classes": 10},
            "cnn": {
                "in_channels": 3 if dataset == "cifar10" else 1,
                "input_size": 32 if dataset == "cifar10" else 28,
                "conv_channels": [32, 64],
                "hidden_dims": [512, 256, 128],
                "num_classes": 10,
            },
        },
        "training": {
            "num_epochs": num_epochs,
            "learning_rate": lr,
            "optimizer": optimizer,
            "seed": SEED,
        },
        "logging": {
            "log_interval": 10,
            "save_dir": str(RESULTS_DIR),
            "save_model": True,
            "save_metrics": True,
            "save_checkpoint": False,
            "verbose": False,
        },
    }
    return config


def run_grid():
    hyperparam_grid = list(
        itertools.product(GRID_LR, GRID_BATCH_SIZES, GRID_EPOCHS, GRID_OPTIMIZERS)
    )
    total_runs = len(ALL_CONFIGS) * len(hyperparam_grid)
    device = get_device()
    criterion = nn.CrossEntropyLoss()

    logger.info("=" * 70)
    logger.info(
        f"FULL GRID SEARCH — {len(ALL_CONFIGS)} configs x {len(hyperparam_grid)} hyperparams = {total_runs} runs"
    )
    logger.info("=" * 70)

    run_counter = 0
    for cfg_idx, (dataset, model_type, use_bn) in enumerate(ALL_CONFIGS, start=1):
        logger.info(
            f"\nConfig [{cfg_idx}/{len(ALL_CONFIGS)}]: {dataset} | {model_type} | bn={use_bn}"
        )

        for lr, batch_size, num_epochs, optimizer in hyperparam_grid:
            run_counter += 1
            logger.info(
                f"  [{run_counter}/{total_runs}] lr={lr} batch={batch_size} epochs={num_epochs} opt={optimizer}"
            )
            config = _build_config(
                dataset, model_type, use_bn, lr, batch_size, num_epochs, optimizer
            )

            t = time.time()
            try:
                model, metrics_tracker, test_loader, _ = train(config)
                test_loss, test_acc = evaluate_final(model, test_loader, criterion, device)
                save_results(model, metrics_tracker, config, test_loss, test_acc)
                metrics = metrics_tracker.get_dict()
                best_val = max(metrics["val_acc"]) if metrics["val_acc"] else 0.0
                logger.info(
                    f"    OK {time.time() - t:.0f}s — val={best_val:.4f} test={test_acc:.4f}"
                )
            except Exception as e:
                logger.error(f"    FAILED {time.time() - t:.0f}s — {e}")
                traceback.print_exc()


# loading


def load_experiments() -> list[dict]:
    experiments = []
    if not RESULTS_DIR.exists():
        return experiments

    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        summary_file = exp_dir / "summary.json"
        if not summary_file.exists():
            continue
        try:
            with open(summary_file) as f:
                summary = json.load(f)
        except Exception:
            continue

        model_file = exp_dir / "model.pt"
        experiments.append(
            {
                "name": exp_dir.name,
                "dir": exp_dir,
                "model_path": model_file if model_file.exists() else None,
                "summary": summary,
            }
        )
    return experiments


def best_for_config(experiments, dataset, model_type, use_bn) -> dict | None:
    filtered = [
        e
        for e in experiments
        if e["summary"].get("dataset") == dataset
        and e["summary"].get("model_type") == model_type
        and e["summary"].get("use_bn") == use_bn
    ]
    if not filtered:
        return None
    return max(filtered, key=lambda e: e["summary"].get("test_acc") or -1.0)


# upload S3


def upload_best_per_config(experiments: list[dict]) -> None:
    s3 = boto3.client("s3")

    print()
    print("=" * 70)
    print(f"UPLOADING BEST MODEL PER CONFIG TO s3://{S3_BUCKET}/{S3_PREFIX}/")
    print("=" * 70)

    for dataset, model_type, use_bn in ALL_CONFIGS:
        best = best_for_config(experiments, dataset, model_type, use_bn)
        if best is None:
            logger.warning(f"No runs found for {dataset}/{model_type}/bn={use_bn} — skipping")
            continue

        s = best["summary"]
        folder = f"{dataset}_{model_type}_bn{use_bn}"
        prefix = f"{S3_PREFIX}/{folder}/"

        print(
            f"\n[{dataset} | {model_type} | bn={use_bn}]  {best['name']}  test_acc={s.get('test_acc', 'N/A')}"
        )

        for filename, local_path in [
            ("model.pt", best["dir"] / "model.pt"),
            ("summary.json", best["dir"] / "summary.json"),
            ("metrics.json", best["dir"] / "metrics.json"),
            ("config.json", best["dir"] / "config.json"),
        ]:
            if not local_path.exists():
                continue
            s3_key = prefix + filename
            try:
                s3.upload_file(str(local_path), S3_BUCKET, s3_key)
                print(f"  OK  {filename:<15} -> s3://{S3_BUCKET}/{s3_key}")
            except (BotoCoreError, ClientError) as e:
                logger.error(f"  FAILED {filename}: {e}")

    print()
    print("=" * 70)
    print("Done.")


def main():
    run_grid()

    experiments = load_experiments()
    if not experiments:
        logger.error("No experiments found after grid search.")
        sys.exit(1)

    upload_best_per_config(experiments)


if __name__ == "__main__":
    main()
