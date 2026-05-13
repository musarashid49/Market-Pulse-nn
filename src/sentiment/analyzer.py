"""
src/sentiment/analyzer.py
==========================
Unified sentiment scoring interface: FinBERT (primary) + VADER (fallback).

FinBERT (ProsusAI/finbert) is pre-trained on financial news and outperforms
general-purpose sentiment models on the FinancialPhraseBank benchmark.
VADER is a fast rule-based model that runs on CPU -- used when FinBERT is
unavailable (no GPU, model download failed, or testing locally).

Both models return a normalised score in [-1, +1]:
  +1 = maximally positive
   0 = neutral
  -1 = maximally negative

Usage:
    from src.sentiment import SentimentAnalyzer
    analyzer = SentimentAnalyzer(device="cuda")   # or "cpu"
    df = analyzer.score(texts=["Apple beats earnings estimates"], method="finbert")
"""

import numpy as np
import pandas as pd
from typing import Literal
from ..utils.helpers import get_logger

logger = get_logger(__name__)

# Label -> numeric score mapping for FinBERT's output classes
FINBERT_LABEL_MAP = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral" :  0.0,
}


class SentimentAnalyzer:
    """
    Scores a list of texts with FinBERT or VADER, returning a DataFrame
    with columns: text, label, score, confidence.

    Parameters
    ----------
    device    : 'cuda' | 'cpu' | 'auto' (auto-detects GPU)
    model_name: HuggingFace model path (default: ProsusAI/finbert)
    """

    def __init__(
        self,
        device: str = "auto",
        model_name: str = "ProsusAI/finbert",
    ) -> None:
        self.model_name = model_name
        self.device     = self._resolve_device(device)
        self._finbert   = None
        self._vader     = None
        logger.info(f"SentimentAnalyzer initialised | device={self.device}")

    # ------------------------------------------------------------------
    def score(
        self,
        texts: list,
        method: Literal["finbert", "vader", "auto"] = "auto",
        batch_size: int = 64,
        max_length: int = 512,
    ) -> pd.DataFrame:
        """
        Scores a list of text strings.

        Parameters
        ----------
        texts      : list of strings to score
        method     : 'finbert' | 'vader' | 'auto'
                     auto = try FinBERT, fall back to VADER on failure
        batch_size : FinBERT batch size (increase on A100: 128)
        max_length : FinBERT max token length (512 is the model limit)

        Returns
        -------
        pd.DataFrame with columns:
          text        : original input
          label       : 'positive' | 'negative' | 'neutral'
          score       : float in [-1, +1]
          confidence  : softmax probability of the winning class (FinBERT)
                        or compound score magnitude (VADER)
          method      : 'finbert' or 'vader'
        """
        if not texts:
            return pd.DataFrame(columns=["text", "label", "score", "confidence", "method"])

        texts = [str(t) for t in texts]

        if method == "finbert" or method == "auto":
            try:
                return self._score_finbert(texts, batch_size, max_length)
            except Exception as exc:
                if method == "finbert":
                    raise
                logger.warning(f"FinBERT failed ({exc}), falling back to VADER.")

        return self._score_vader(texts)

    # ------------------------------------------------------------------
    def _score_finbert(
        self, texts: list, batch_size: int, max_length: int
    ) -> pd.DataFrame:
        """Scores texts using FinBERT via the HuggingFace pipeline."""
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

        if self._finbert is None:
            logger.info(f"Loading FinBERT ({self.model_name}) onto {self.device} ...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model     = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._finbert = pipeline(
                task      = "text-classification",
                model     = model,
                tokenizer = tokenizer,
                device    = 0 if self.device == "cuda" else -1,
                top_k     = None,       # return all class probabilities
            )
            logger.info("FinBERT loaded.")

        # Truncate long texts at the token level (FinBERT max = 512)
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # pipeline handles tokenisation + inference
            outputs = self._finbert(
                batch,
                truncation  = True,
                max_length  = max_length,
                padding     = True,
                batch_size  = min(batch_size, len(batch)),
            )
            results.extend(outputs)

        rows = []
        for text, output in zip(texts, results):
            # output is a list of {label, score} dicts (all classes)
            best     = max(output, key=lambda x: x["score"])
            label    = best["label"].lower()
            conf     = best["score"]
            # Weighted score: positive - negative (neutral contributes 0)
            p = next((x["score"] for x in output if x["label"].lower() == "positive"), 0.0)
            n = next((x["score"] for x in output if x["label"].lower() == "negative"), 0.0)
            numeric_score = p - n  # range [-1, +1]

            rows.append({
                "text"      : text,
                "label"     : label,
                "score"     : numeric_score,
                "confidence": conf,
                "method"    : "finbert",
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def _score_vader(self, texts: list) -> pd.DataFrame:
        """Scores texts using VADER (CPU, no model download needed)."""
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        if self._vader is None:
            self._vader = SentimentIntensityAnalyzer()
            logger.info("VADER analyser initialised.")

        pos_thresh = 0.05
        neg_thresh = -0.05

        rows = []
        for text in texts:
            scores = self._vader.polarity_scores(text)
            compound = scores["compound"]  # already in [-1, +1]

            if compound >= pos_thresh:
                label = "positive"
            elif compound <= neg_thresh:
                label = "negative"
            else:
                label = "neutral"

            rows.append({
                "text"      : text,
                "label"     : label,
                "score"     : compound,
                "confidence": abs(compound),   # proxy for confidence
                "method"    : "vader",
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def aggregate_daily(
        self,
        df: pd.DataFrame,
        date_col: str = "published",
        score_col: str = "score",
    ) -> pd.DataFrame:
        """
        Aggregates scored records into daily sentiment features.

        Returns a DataFrame indexed by date with columns:
          sentiment_mean, sentiment_std, sentiment_pos_ratio,
          sentiment_neg_ratio, record_count
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df[date_col], utc=True).dt.date

        agg = df.groupby("date").agg(
            sentiment_mean    = (score_col, "mean"),
            sentiment_std     = (score_col, "std"),
            record_count      = (score_col, "count"),
        ).reset_index()

        # Positive/negative ratios
        pos_counts = (
            df[df["label"] == "positive"].groupby("date")[score_col].count()
        )
        neg_counts = (
            df[df["label"] == "negative"].groupby("date")[score_col].count()
        )
        agg = agg.set_index("date")
        agg["sentiment_pos_ratio"] = (pos_counts / agg["record_count"]).fillna(0)
        agg["sentiment_neg_ratio"] = (neg_counts / agg["record_count"]).fillna(0)
        agg["date"] = pd.to_datetime(agg.index)

        return agg.reset_index(drop=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device
