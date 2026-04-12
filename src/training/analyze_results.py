"""
Analysis and visualization script for training results.

Usage:
    python src/training/analyze_results.py
    python src/training/analyze_results.py --results-dir ./results
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyze and visualize training results."""

    def __init__(self, results_dir: str = "./results"):
        """Initialize analyzer."""
        self.results_dir = Path(results_dir)
        self.experiments = {}
        self.load_experiments()

    def load_experiments(self) -> None:
        """Load all experiment results from results directory."""
        if not self.results_dir.exists():
            logger.warning(f"Results directory {self.results_dir} not found")
            return

        for exp_dir in sorted(self.results_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            metrics_file = exp_dir / "metrics.json"
            summary_file = exp_dir / "summary.json"

            if metrics_file.exists() and summary_file.exists():
                try:
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                    with open(summary_file) as f:
                        summary = json.load(f)

                    self.experiments[exp_dir.name] = {
                        "dir": exp_dir,
                        "metrics": metrics,
                        "summary": summary,
                    }
                    logger.info(f"Loaded experiment: {exp_dir.name}")
                except Exception as e:
                    logger.error(f"Error loading {exp_dir.name}: {e}")

    def print_summaries(self) -> None:
        """Print summaries of all experiments."""
        if not self.experiments:
            logger.warning("No experiments found")
            return

        print("\n" + "=" * 100)
        print("EXPERIMENT SUMMARIES")
        print("=" * 100)

        for exp_name, exp_data in sorted(self.experiments.items()):
            summary = exp_data["summary"]
            config = summary.get("config", {})

            print(f"\n{exp_name}")
            print("-" * 100)
            print(f"  Dataset: {config.get('dataset', {}).get('name', 'N/A')}")
            print(f"  Model: {config.get('model', {}).get('type', 'N/A')}")
            print(f"  BatchNorm: {config.get('model', {}).get('use_bn', 'N/A')}")
            print(f"  Optimizer: {config.get('training', {}).get('optimizer', 'N/A')}")
            print(f"  Learning Rate: {config.get('training', {}).get('learning_rate', 'N/A')}")
            print(f"  Final Train Loss: {summary.get('final_train_loss', 'N/A'):.6f}")
            print(f"  Final Train Acc: {summary.get('final_train_acc', 'N/A'):.4f}")
            print(f"  Final Val Loss: {summary.get('final_val_loss', 'N/A'):.6f}")
            print(f"  Final Val Acc: {summary.get('final_val_acc', 'N/A'):.4f}")
            print(f"  Best Val Acc: {summary.get('best_val_acc', 'N/A'):.4f}")

    def compare_with_without_bn(self) -> None:
        """Compare results with and without BatchNorm."""
        print("\n" + "=" * 100)
        print("BATCHNORM COMPARISON")
        print("=" * 100)

        # Group by dataset and model
        groups: dict[str, list[tuple]] = {}

        for _exp_name, exp_data in self.experiments.items():
            config = exp_data["summary"].get("config", {})
            dataset = config.get("dataset", {}).get("name")
            model_type = config.get("model", {}).get("type")
            use_bn = config.get("model", {}).get("use_bn")

            key = f"{dataset}_{model_type}"
            if key not in groups:
                groups[key] = []

            groups[key].append((use_bn, exp_data))

        for key, experiments in sorted(groups.items()):
            print(f"\n{key.upper()}")
            print("-" * 100)

            # Sort by use_bn
            experiments_sorted = sorted(experiments, key=lambda x: x[0])

            for use_bn, exp_data in experiments_sorted:
                summary = exp_data["summary"]
                bn_str = "WITH" if use_bn else "WITHOUT"
                print(f"  {bn_str} BatchNorm:")
                print(f"    Final Val Acc: {summary.get('final_val_acc', 0):.4f}")
                print(f"    Best Val Acc: {summary.get('best_val_acc', 0):.4f}")
                print(f"    Final Val Loss: {summary.get('final_val_loss', 0):.6f}")

    def plot_metrics(self, figsize: tuple = (15, 10)) -> None:
        """Plot metrics for all experiments."""
        if not self.experiments:
            logger.warning("No experiments to plot")
            return

        num_exps = len(self.experiments)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("Training Metrics Across All Experiments", fontsize=16, fontweight="bold")

        colors = plt.cm.Set3(np.linspace(0, 1, num_exps))

        for idx, (_exp_name, exp_data) in enumerate(sorted(self.experiments.items())):
            metrics = exp_data["metrics"]
            summary = exp_data["summary"]

            config = summary.get("config", {})
            model_type = config.get("model", {}).get("type")
            use_bn = config.get("model", {}).get("use_bn")
            dataset = config.get("dataset", {}).get("name")

            label = f"{dataset[:4]}_{model_type[0]}_bn{str(use_bn)[0]}"

            # Créer l'axe X avec epochs numérotés à partir de 1
            if metrics.get("train_loss"):
                epochs = np.arange(1, len(metrics["train_loss"]) + 1)

                # Training Loss
                axes[0, 0].plot(
                    epochs,
                    metrics["train_loss"],
                    label=label,
                    color=colors[idx],
                    linestyle="-",
                    linewidth=1.5,
                )

                # Validation Loss
                if "val_loss" in metrics:
                    axes[0, 1].plot(
                        epochs,
                        metrics["val_loss"],
                        label=label,
                        color=colors[idx],
                        linestyle="-",
                        linewidth=1.5,
                    )

                # Training Accuracy
                if "train_acc" in metrics:
                    axes[1, 0].plot(
                        epochs,
                        metrics["train_acc"],
                        label=label,
                        color=colors[idx],
                        linestyle="-",
                        linewidth=1.5,
                    )

                # Validation Accuracy
                if "val_acc" in metrics:
                    axes[1, 1].plot(
                        epochs,
                        metrics["val_acc"],
                        label=label,
                        color=colors[idx],
                        linestyle="-",
                        linewidth=1.5,
                    )

        # Configure subplots
        axes[0, 0].set_title("Training Loss", fontweight="bold")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend(loc="best", fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_title("Validation Loss", fontweight="bold")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend(loc="best", fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set_title("Training Accuracy", fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].legend(loc="best", fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_title("Validation Accuracy", fontweight="bold")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Accuracy")
        axes[1, 1].legend(loc="best", fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.results_dir / "comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved comparison plot to {output_path}")
        plt.show()

    def generate_report(self) -> str:
        """Generate a text report of all results."""
        report = []
        report.append("=" * 100)
        report.append("TRAINING RESULTS REPORT")
        report.append("=" * 100)
        report.append("")

        report.append(f"Total Experiments: {len(self.experiments)}")
        report.append("")

        # Summary statistics
        if self.experiments:
            val_accs = [exp["summary"].get("best_val_acc", 0) for exp in self.experiments.values()]
            report.append(f"Best Val Acc: {max(val_accs):.4f}")
            report.append(f"Mean Val Acc: {np.mean(val_accs):.4f}")
            report.append(f"Std Val Acc: {np.std(val_accs):.4f}")
            report.append("")

        # Detailed results
        for exp_name, exp_data in sorted(self.experiments.items()):
            summary = exp_data["summary"]
            config = summary.get("config", {})

            report.append(f"\n{exp_name}")
            report.append("-" * 100)
            report.append(f"  Dataset: {config.get('dataset', {}).get('name')}")
            report.append(f"  Model: {config.get('model', {}).get('type')}")
            report.append(f"  BatchNorm: {config.get('model', {}).get('use_bn')}")
            report.append(f"  LR: {config.get('training', {}).get('learning_rate')}")
            report.append(f"  Final Train Loss: {summary.get('final_train_loss', 0):.6f}")
            report.append(f"  Final Val Loss: {summary.get('final_val_loss', 0):.6f}")
            report.append(f"  Final Train Acc: {summary.get('final_train_acc', 0):.4f}")
            report.append(f"  Final Val Acc: {summary.get('final_val_acc', 0):.4f}")
            report.append(f"  Best Val Acc: {summary.get('best_val_acc', 0):.4f}")

        report_text = "\n".join(report)

        # Save report
        report_path = self.results_dir / "report.txt"
        with open(report_path, "w") as f:
            f.write(report_text)
        logger.info(f"Saved report to {report_path}")

        return report_text


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze training results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate text report",
    )

    args = parser.parse_args()

    analyzer = ResultsAnalyzer(args.results_dir)

    # Print summaries
    analyzer.print_summaries()

    # Compare with/without BN
    analyzer.compare_with_without_bn()

    # Generate report
    if args.report:
        report = analyzer.generate_report()
        print("\n" + report)

    # Plot metrics
    if args.plot:
        analyzer.plot_metrics()


if __name__ == "__main__":
    main()
