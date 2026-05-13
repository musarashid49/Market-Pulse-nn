"""
src/utils/helpers.py
====================
Project-wide utility functions shared across all notebooks and modules.

Covers:
  - Reproducibility: fix all random seeds (Python, NumPy, PyTorch, CUDA)
  - Logging: structured console logger with timestamps
  - Data validation: schema checks before feeding to models
  - Device detection: returns the best available device
"""

import os
import sys
import random
import logging
from pathlib import Path
from datetime import datetime

import numpy as np


# ─── Logging ─────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a named logger that writes timestamped messages to stdout.
    Use one logger per module, e.g.:
        logger = get_logger(__name__)
        logger.info("Loaded 1200 rows")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Fixes all random seeds for full reproducibility across:
      Python random, NumPy, PyTorch CPU, PyTorch CUDA.
    Call this at the top of every notebook before any data splits or model inits.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Deterministic operations (slightly slower, but reproducible)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    print(f"[set_seed] All seeds fixed to {seed}")


# ─── Device Detection ─────────────────────────────────────────────────────────

def get_device():
    """
    Returns the best available torch device.
    On Colab A100 this will return 'cuda'.
    Prints a human-readable summary including GPU name if available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[get_device] Using GPU: {gpu_name} ({mem_gb:.1f} GB)")
        else:
            device = torch.device("cpu")
            print("[get_device] No GPU found — using CPU")
        return device
    except ImportError:
        print("[get_device] PyTorch not found — returning 'cpu' string")
        return "cpu"


# ─── Data Validation ─────────────────────────────────────────────────────────

def validate_dataframe(df, required_cols: list[str], name: str = "DataFrame") -> bool:
    """
    Checks that a DataFrame has all required columns and no fully-null columns.
    Raises ValueError if validation fails so notebooks catch it early.

    Example:
        validate_dataframe(df, ["date", "close", "sentiment_score"], "price_sentiment_df")
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[validate_dataframe] {name} is missing columns: {missing}")

    null_cols = [c for c in df.columns if df[c].isna().all()]
    if null_cols:
        raise ValueError(f"[validate_dataframe] {name} has entirely-null columns: {null_cols}")

    print(f"[validate_dataframe] ✓ {name} — shape={df.shape}, cols={list(df.columns)}")
    return True


def check_data_leakage(train_index, test_index) -> bool:
    """
    Verifies that the train and test index ranges do not overlap.
    Critical: time-series splits must be strictly chronological.
    """
    train_set = set(train_index)
    test_set  = set(test_index)
    overlap   = train_set.intersection(test_set)

    if overlap:
        raise ValueError(
            f"[check_data_leakage] DATA LEAKAGE DETECTED: "
            f"{len(overlap)} indices appear in both train and test sets!"
        )

    if max(train_index) >= min(test_index):
        raise ValueError(
            f"[check_data_leakage] Train max index ({max(train_index)}) "
            f">= test min index ({min(test_index)}). "
            "Train data must come strictly before test data."
        )

    print(
        f"[check_data_leakage] ✓ No leakage — "
        f"train ends at {max(train_index)}, test starts at {min(test_index)}"
    )
    return True


# ─── File I/O helpers ─────────────────────────────────────────────────────────

def timestamp_filename(prefix: str, ext: str = "csv") -> str:
    """
    Generates a timestamped filename to avoid overwriting previous runs.
    Example: timestamp_filename("prices") → "prices_20240601_143022.csv"
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{ext}"


def ensure_dir(path: Path) -> Path:
    """Creates directory (and parents) if it doesn't exist. Returns path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
