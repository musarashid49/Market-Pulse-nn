"""
src/models/trainer.py
======================
Shared training loop used by all three models (RNN, LSTM, GRU).

Features:
  - BCEWithLogitsLoss for binary direction classification
  - Adam optimiser with configurable learning rate and weight decay
  - Cosine annealing LR scheduler (warm restarts optional)
  - Early stopping on validation F1-score
  - Checkpoint saving (best val F1 model)
  - Per-epoch metric logging returned as a DataFrame

Usage:
    from src.models import Trainer, LSTMModel

    model   = LSTMModel(input_size=n_features)
    trainer = Trainer(model=model, config=CFG, device=device)
    history = trainer.train(train_dl, val_dl)
    trainer.load_best()   # restore best checkpoint
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, mean_squared_error,
)
from ..utils.helpers import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """
    Stops training when the monitored metric stops improving.
    Saves the best model state to disk.
    """

    def __init__(self, patience: int, save_path: Path, mode: str = "max") -> None:
        self.patience   = patience
        self.save_path  = save_path
        self.mode       = mode
        self.best_score: Optional[float] = None
        self.counter    = 0
        self.stop       = False

    def __call__(self, score: float, model: nn.Module) -> None:
        improved = (
            self.best_score is None
            or (self.mode == "max" and score > self.best_score)
            or (self.mode == "min" and score < self.best_score)
        )
        if improved:
            self.best_score = score
            torch.save(model.state_dict(), self.save_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Manages the full training + validation loop for a single model.

    Parameters
    ----------
    model  : nn.Module -- VanillaRNN, LSTMModel, or GRUModel
    config : CFG object with hyperparameters
    device : torch.device
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        device: torch.device,
    ) -> None:
        self.model  = model.to(device)
        self.cfg    = config
        self.device = device

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr           = config.LEARNING_RATE,
            weight_decay = config.WEIGHT_DECAY,
        )

        # Cosine annealing: smoothly reduces LR from max to near-zero each cycle
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.EPOCHS, eta_min=1e-6
        )

        # Checkpoint path: unique per model name
        model_name = getattr(model, "model_name", model.__class__.__name__)
        ckpt_path  = Path(config.MODELS) / f"{model_name}_best.pt"

        self.early_stop = EarlyStopping(
            patience  = config.PATIENCE,
            save_path = ckpt_path,
            mode      = "max",   # maximise validation F1
        )
        self.ckpt_path = ckpt_path
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> pd.DataFrame:
        """
        Runs the training loop for up to CFG.EPOCHS epochs.

        Returns
        -------
        pd.DataFrame with per-epoch metrics (train_loss, val_loss,
        val_acc, val_f1, val_auc, lr, epoch_time_s)
        """
        model_name = getattr(self.model, "model_name", self.model.__class__.__name__)
        logger.info(
            f"Training {model_name} | epochs={self.cfg.EPOCHS} | "
            f"patience={self.cfg.PATIENCE} | device={self.device}"
        )

        for epoch in range(1, self.cfg.EPOCHS + 1):
            t0 = time.time()

            train_loss = self._train_epoch(train_loader)
            val_metrics = self._eval_epoch(val_loader)

            elapsed = time.time() - t0
            lr      = self.optimizer.param_groups[0]["lr"]

            row = {
                "epoch"        : epoch,
                "train_loss"   : round(train_loss, 5),
                "val_loss"     : round(val_metrics["loss"], 5),
                "val_acc"      : round(val_metrics["acc"],  4),
                "val_f1"       : round(val_metrics["f1"],   4),
                "val_auc"      : round(val_metrics["auc"],  4),
                "lr"           : lr,
                "epoch_time_s" : round(elapsed, 2),
            }
            self.history.append(row)

            logger.info(
                f"Ep {epoch:03d}/{self.cfg.EPOCHS} | "
                f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | val_acc={val_metrics['acc']:.4f} | "
                f"lr={lr:.2e} | {elapsed:.1f}s"
            )

            self.early_stop(val_metrics["f1"], self.model)
            self.scheduler.step()

            if self.early_stop.stop:
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best val F1: {self.early_stop.best_score:.4f}"
                )
                break

        return pd.DataFrame(self.history)

    # ------------------------------------------------------------------
    def _train_epoch(self, loader: DataLoader) -> float:
        """Single training epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss   = self.criterion(logits, y_batch)
            loss.backward()

            # Gradient clipping prevents exploding gradients in RNNs
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item() * len(X_batch)

        return total_loss / len(loader.dataset)

    # ------------------------------------------------------------------
    def _eval_epoch(self, loader: DataLoader) -> dict:
        """Single evaluation epoch. Returns loss and classification metrics."""
        self.model.eval()
        total_loss = 0.0
        all_probs:  list = []
        all_labels: list = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(X_batch)
                loss   = self.criterion(logits, y_batch)
                total_loss += loss.item() * len(X_batch)

                probs  = torch.sigmoid(logits).cpu().numpy()
                labels = y_batch.cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels)

        all_probs  = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds      = (all_probs >= 0.5).astype(int)

        return {
            "loss": total_loss / len(loader.dataset),
            "acc" : accuracy_score(all_labels, preds),
            "f1"  : f1_score(all_labels, preds, average="macro", zero_division=0),
            "auc" : roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5,
            "rmse": mean_squared_error(all_labels, all_probs, squared=False),
        }

    # ------------------------------------------------------------------
    def evaluate(self, loader: DataLoader) -> dict:
        """Public evaluation method -- use after training on the test set."""
        return self._eval_epoch(loader)

    # ------------------------------------------------------------------
    def load_best(self) -> None:
        """Restores the best checkpoint saved during training."""
        if self.ckpt_path.exists():
            self.model.load_state_dict(
                torch.load(self.ckpt_path, map_location=self.device)
            )
            logger.info(f"Best checkpoint loaded from {self.ckpt_path.name}")
        else:
            logger.warning(f"No checkpoint found at {self.ckpt_path}")

    # ------------------------------------------------------------------
    def predict_proba(self, loader: DataLoader) -> np.ndarray:
        """Returns predicted probabilities for a DataLoader."""
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                probs   = torch.sigmoid(self.model(X_batch)).cpu().numpy()
                all_probs.extend(probs)
        return np.array(all_probs)
