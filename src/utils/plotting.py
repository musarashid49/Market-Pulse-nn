"""
src/utils/plotting.py
======================
Centralised visualisation functions for the market-pulse-nn project.

All functions accept a `save_path` argument. When provided, the figure
is saved as a PNG at 150 DPI before display. Pass None to show only.

Available plots:
  Plotter.training_curves     -- loss + F1/acc over epochs for one model
  Plotter.compare_models      -- side-by-side metric curves for all three models
  Plotter.confusion_matrix    -- normalised confusion matrix heatmap
  Plotter.roc_curve           -- ROC curve with AUC annotation
  Plotter.prediction_vs_actual-- predicted probability vs true direction over time
  Plotter.model_comparison_table -- final metric comparison bar chart
  Plotter.feature_importance  -- horizontal bar chart of feature weights
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Optional
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    classification_report,
)
from .helpers import get_logger

logger = get_logger(__name__)


class Plotter:
    """
    Static-method collection for all project visualisations.
    Instantiate once with a figures directory and call methods on the instance.

    Usage:
        plotter = Plotter(figures_dir=CFG.FIGURES)
        plotter.training_curves(history_df, model_name="LSTM")
    """

    MODEL_COLORS = {
        "VanillaRNN": "#E65100",
        "LSTM"      : "#1565C0",
        "GRU"       : "#2E7D32",
    }

    def __init__(self, figures_dir: Path) -> None:
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def training_curves(
        self,
        history: pd.DataFrame,
        model_name: str,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plots train/val loss and val F1+acc over training epochs.

        Parameters
        ----------
        history    : DataFrame returned by Trainer.train()
        model_name : used in title and auto-generated filename
        """
        fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"{model_name} -- Training History", fontsize=14, fontweight="bold")

        color = self.MODEL_COLORS.get(model_name, "#333")

        # ── Loss curves ────────────────────────────────────────────────
        ax_loss.plot(history["epoch"], history["train_loss"],
                     label="Train loss", color=color, linewidth=1.8)
        ax_loss.plot(history["epoch"], history["val_loss"],
                     label="Val loss", color=color, linestyle="--", linewidth=1.8, alpha=0.7)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("BCE Loss")
        ax_loss.set_title("Loss (train vs validation)")
        ax_loss.legend()

        # Mark best val loss
        best_epoch = history["val_f1"].idxmax()
        ax_loss.axvline(history.loc[best_epoch, "epoch"], color="#90A4AE",
                        linestyle=":", alpha=0.7, label="Best F1 epoch")

        # ── Metric curves ───────────────────────────────────────────────
        ax_metric.plot(history["epoch"], history["val_f1"],
                       label="Val F1",  color=color, linewidth=2.0)
        ax_metric.plot(history["epoch"], history["val_acc"],
                       label="Val Acc", color=color, linestyle="--",
                       linewidth=1.6, alpha=0.75)
        if "val_auc" in history.columns:
            ax_metric.plot(history["epoch"], history["val_auc"],
                           label="Val AUC", color=color, linestyle=":",
                           linewidth=1.4, alpha=0.6)
        ax_metric.set_xlabel("Epoch")
        ax_metric.set_ylabel("Score")
        ax_metric.set_title("Validation metrics over training")
        ax_metric.set_ylim(0, 1)
        ax_metric.legend()

        plt.tight_layout()
        out = save_path or self.figures_dir / f"05_{model_name.lower()}_training_curves.png"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.show()
        logger.info(f"Saved {out.name}")

    # ------------------------------------------------------------------
    def compare_models(
        self,
        histories: dict,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Overlays val F1 and val loss curves for all three models.

        Parameters
        ----------
        histories : dict[model_name -> history_df]
        """
        fig, (ax_f1, ax_loss) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Model Comparison -- Training Curves", fontsize=14, fontweight="bold")

        for model_name, history in histories.items():
            color = self.MODEL_COLORS.get(model_name, "#888")
            ax_f1.plot(history["epoch"], history["val_f1"],
                       label=model_name, color=color, linewidth=1.9)
            ax_loss.plot(history["epoch"], history["val_loss"],
                         label=model_name, color=color, linewidth=1.9)

        for ax, ylabel, title in [
            (ax_f1,   "Val F1 (macro)",  "Validation F1 -- all models"),
            (ax_loss, "Val BCE Loss",     "Validation Loss -- all models"),
        ]:
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()

        plt.tight_layout()
        out = save_path or self.figures_dir / "05_model_comparison_curves.png"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.show()
        logger.info(f"Saved {out.name}")

    # ------------------------------------------------------------------
    def confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        save_path: Optional[Path] = None,
    ) -> None:
        """Normalised confusion matrix with class labels."""
        cm   = confusion_matrix(y_true, y_pred, normalize="true")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["Down (0)", "Up (1)"],
            yticklabels=["Down (0)", "Up (1)"],
            ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8},
        )
        ax.set_title(f"{model_name} -- Confusion Matrix (normalised)")
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        plt.tight_layout()
        out = save_path or self.figures_dir / f"05_{model_name.lower()}_confusion_matrix.png"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.show()
        logger.info(f"Saved {out.name}")

    # ------------------------------------------------------------------
    def roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str,
        ax: Optional[plt.Axes] = None,
        save_path: Optional[Path] = None,
    ) -> None:
        """ROC curve with AUC annotation. Can plot onto an existing ax."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc     = auc(fpr, tpr)
        color       = self.MODEL_COLORS.get(model_name, "#333")

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(6, 5))

        ax.plot(fpr, tpr, color=color, linewidth=2.0,
                label=f"{model_name}  AUC={roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")

        if standalone:
            plt.tight_layout()
            out = save_path or self.figures_dir / f"05_{model_name.lower()}_roc.png"
            plt.savefig(out, bbox_inches="tight", dpi=150)
            plt.show()
            logger.info(f"Saved {out.name}")

    # ------------------------------------------------------------------
    def roc_all_models(
        self,
        results: dict,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plots ROC curves for all models on one axis.

        Parameters
        ----------
        results : dict[model_name -> {"y_true": ..., "y_prob": ...}]
        """
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.suptitle("ROC Curves -- All Models", fontsize=13, fontweight="bold")

        for model_name, res in results.items():
            self.roc_curve(res["y_true"], res["y_prob"], model_name, ax=ax)

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
        ax.legend(loc="lower right")
        plt.tight_layout()
        out = save_path or self.figures_dir / "05_roc_all_models.png"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.show()
        logger.info(f"Saved {out.name}")

    # ------------------------------------------------------------------
    def prediction_vs_actual(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str,
        dates: Optional[pd.DatetimeIndex] = None,
        n_samples: int = 200,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plots predicted probability vs true direction over time.
        Provides intuition for how well the model tracks the true signal.
        """
        idx   = np.arange(min(n_samples, len(y_true)))
        x_axis = dates[idx] if dates is not None else idx
        color = self.MODEL_COLORS.get(model_name, "#333")

        fig, (ax_prob, ax_dir) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
        fig.suptitle(f"{model_name} -- Predictions vs Actuals (first {n_samples} test samples)",
                     fontsize=13, fontweight="bold")

        # Predicted probability
        ax_prob.plot(x_axis, y_prob[idx], color=color, linewidth=1.3, alpha=0.8, label="Predicted prob")
        ax_prob.axhline(0.5, color="#90A4AE", linestyle="--", linewidth=0.8)
        ax_prob.set_ylabel("P(Up)")
        ax_prob.set_title("Predicted probability of up-move")
        ax_prob.set_ylim(0, 1)
        ax_prob.legend()

        # True direction as a step function
        ax_dir.step(x_axis, y_true[idx], where="post",
                    color="#1B5E20", linewidth=1.2, alpha=0.9, label="True direction")
        ax_dir.fill_between(x_axis, 0, y_true[idx],
                            step="post", alpha=0.15, color="#1B5E20")
        ax_dir.set_ylabel("Direction (1=Up, 0=Down)")
        ax_dir.set_xlabel("Sample index" if dates is None else "Date")
        ax_dir.set_title("True next-day direction")
        ax_dir.legend()

        plt.tight_layout()
        out = save_path or self.figures_dir / f"05_{model_name.lower()}_pred_vs_actual.png"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.show()
        logger.info(f"Saved {out.name}")

    # ------------------------------------------------------------------
    def model_comparison_bar(
        self,
        metrics_df: pd.DataFrame,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Grouped bar chart comparing Accuracy, F1, AUC across models.

        Parameters
        ----------
        metrics_df : DataFrame with columns [model, accuracy, f1, auc, rmse]
        """
        metrics = ["accuracy", "f1", "auc"]
        x       = np.arange(len(metrics))
        width   = 0.22
        n       = len(metrics_df)

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle("Final Test Metrics -- Model Comparison", fontsize=14, fontweight="bold")

        for i, (_, row) in enumerate(metrics_df.iterrows()):
            color  = self.MODEL_COLORS.get(row["model"], "#888")
            offset = (i - n / 2 + 0.5) * width
            bars   = ax.bar(
                x + offset,
                [row[m] for m in metrics],
                width, label=row["model"], color=color, alpha=0.85,
            )
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=9,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title("Higher is better for all metrics shown")

        plt.tight_layout()
        out = save_path or self.figures_dir / "05_final_metrics_comparison.png"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.show()
        logger.info(f"Saved {out.name}")
