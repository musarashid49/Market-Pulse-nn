"""
src/features/timeseries.py
===========================
Sliding-window sequence construction for sequential model training.

Key design decisions:
  - Splits are CHRONOLOGICAL (no shuffling before split) to prevent
    look-ahead bias -- a critical requirement in financial ML.
  - Scaler is fit on TRAINING set only, then applied to val/test.
    Fitting on the full dataset would leak future statistics into training.
  - The PyTorch Dataset class works seamlessly with DataLoader for
    batched training with proper shuffling within the train set.

Usage:
    from src.features import SequenceBuilder, MarketDataset

    builder = SequenceBuilder(config=CFG)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = \
        builder.build(df, feature_cols=FEATURE_COLS)

    train_ds = MarketDataset(X_train, y_train)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import pickle

from sklearn.preprocessing import MinMaxScaler
from ..utils.helpers import get_logger, check_data_leakage

logger = get_logger(__name__)


class SequenceBuilder:
    """
    Converts a flat feature DataFrame into (X, y) sequence arrays
    using a sliding window, then splits chronologically.

    Parameters from CFG:
      LOOKBACK, STRIDE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    """

    def __init__(self, config) -> None:
        self.cfg = config

    # ------------------------------------------------------------------
    def build(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        target_col: str = "direction",
        save_scaler_path: Optional[Path] = None,
    ) -> tuple:
        """
        Full pipeline: scale -> window -> split.

        Parameters
        ----------
        df               : feature DataFrame sorted by date (ascending)
        feature_cols     : list of column names to use as model inputs
        target_col       : name of the binary target column
        save_scaler_path : if provided, saves the fitted scaler to disk

        Returns
        -------
        X_train, y_train, X_val, y_val, X_test, y_test, scaler
        All X arrays have shape (n_samples, lookback, n_features).
        All y arrays have shape (n_samples,).
        """
        df = df.sort_values("date").reset_index(drop=True)

        # Step 1: Chronological split BEFORE scaling to prevent leakage
        n      = len(df)
        n_train = int(n * self.cfg.TRAIN_RATIO)
        n_val   = int(n * self.cfg.VAL_RATIO)

        train_df = df.iloc[:n_train]
        val_df   = df.iloc[n_train : n_train + n_val]
        test_df  = df.iloc[n_train + n_val:]

        logger.info(
            f"Chronological split | train: {len(train_df):,} | "
            f"val: {len(val_df):,} | test: {len(test_df):,}"
        )

        # Step 2: Fit scaler on TRAIN ONLY, transform all splits
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_df[feature_cols])
        val_scaled   = scaler.transform(val_df[feature_cols])
        test_scaled  = scaler.transform(test_df[feature_cols])

        if save_scaler_path is not None:
            with open(save_scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            logger.info(f"Scaler saved to {save_scaler_path}")

        # Step 3: Build sliding windows
        X_train, y_train = self._make_windows(train_scaled, train_df[target_col].values)
        X_val,   y_val   = self._make_windows(val_scaled,   val_df[target_col].values)
        X_test,  y_test  = self._make_windows(test_scaled,  test_df[target_col].values)

        logger.info(
            f"Sequence shapes | X_train: {X_train.shape} | "
            f"X_val: {X_val.shape} | X_test: {X_test.shape}"
        )

        return X_train, y_train, X_val, y_val, X_test, y_test, scaler

    # ------------------------------------------------------------------
    def _make_windows(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> tuple:
        """
        Creates overlapping windows from a 2D feature array.

        Each sample X[i] = features[i : i+lookback]   shape (lookback, n_features)
        Label     y[i] = targets[i + lookback]         scalar

        Parameters
        ----------
        features : (n_rows, n_features) scaled array
        targets  : (n_rows,) target labels aligned to features

        Returns
        -------
        X : (n_samples, lookback, n_features)
        y : (n_samples,)
        """
        lookback = self.cfg.LOOKBACK
        stride   = self.cfg.STRIDE
        X, y     = [], []

        for i in range(0, len(features) - lookback, stride):
            X.append(features[i : i + lookback])
            y.append(targets[i + lookback])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def load_scaler(path: Path) -> MinMaxScaler:
        """Loads a previously saved scaler from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────

class MarketDataset:
    """
    PyTorch-compatible Dataset wrapping (X, y) numpy arrays.

    Usage:
        from torch.utils.data import DataLoader
        train_ds  = MarketDataset(X_train, y_train)
        train_dl  = DataLoader(train_ds, batch_size=64, shuffle=True)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        import torch
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
