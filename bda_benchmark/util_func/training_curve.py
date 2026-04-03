import csv
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrainingCurveRecorder:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.loss_history: List[Dict[str, float]] = []
        self.metric_history: List[Dict[str, float]] = []

    def add_train_loss(self, iteration: int, loss: float) -> None:
        self.loss_history.append({"iter": int(iteration), "loss": float(loss)})

    def add_eval_metrics(self, iteration: int, split: str, oa: float, miou: float) -> None:
        self.metric_history.append(
            {
                "iter": int(iteration),
                "split": split,
                "oa": float(oa),
                "miou": float(miou),
            }
        )

    def save(self) -> None:
        if not self.loss_history and not self.metric_history:
            return

        os.makedirs(self.save_dir, exist_ok=True)
        self._save_csv()
        self._save_plot()

    def _save_csv(self) -> None:
        loss_csv = os.path.join(self.save_dir, "train_loss_curve.csv")
        with open(loss_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["iter", "loss"])
            writer.writeheader()
            writer.writerows(self.loss_history)

        metric_csv = os.path.join(self.save_dir, "eval_metric_curve.csv")
        with open(metric_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["iter", "split", "oa", "miou"])
            writer.writeheader()
            writer.writerows(self.metric_history)

    def _save_plot(self) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        if self.loss_history:
            loss_iters = [x["iter"] for x in self.loss_history]
            loss_vals = [x["loss"] for x in self.loss_history]
            axes[0].plot(loss_iters, loss_vals, color="#e66a19", linewidth=1.6)
            axes[0].set_title("Train Loss")
            axes[0].set_xlabel("Iteration")
            axes[0].set_ylabel("Loss")
            axes[0].grid(alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, "No train loss data", ha="center", va="center")
            axes[0].set_axis_off()

        split_styles = {
            "val": {"oa_color": "#1f77b4", "miou_color": "#2ca02c"},
            "test": {"oa_color": "#9467bd", "miou_color": "#d62728"},
        }

        has_metric = False
        for split, style in split_styles.items():
            split_data = [x for x in self.metric_history if x["split"] == split]
            if not split_data:
                continue
            has_metric = True
            x_axis = [x["iter"] for x in split_data]
            oa_vals = [x["oa"] for x in split_data]
            miou_vals = [x["miou"] for x in split_data]
            axes[1].plot(
                x_axis,
                oa_vals,
                marker="o",
                markersize=3,
                linewidth=1.4,
                color=style["oa_color"],
                label=f"{split.upper()} OA",
            )
            axes[1].plot(
                x_axis,
                miou_vals,
                marker="s",
                markersize=3,
                linewidth=1.4,
                color=style["miou_color"],
                label=f"{split.upper()} mIoU",
            )

        if has_metric:
            axes[1].set_title("Validation/Test Metrics")
            axes[1].set_xlabel("Iteration")
            axes[1].set_ylabel("Score (%)")
            axes[1].grid(alpha=0.3)
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, "No eval metric data", ha="center", va="center")
            axes[1].set_axis_off()

        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, "training_curve.png"), dpi=200)
        plt.close(fig)
