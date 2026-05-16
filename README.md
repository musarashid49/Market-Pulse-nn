# market-pulse-nn

**Real-Time Market Movement Prediction using Sequential Deep Learning**

End-to-end pipeline that fuses Yahoo Finance price data with multi-source
financial news sentiment (Alpha Vantage, NewsData.io, Reuters RSS) and trains
three sequential deep learning models (Vanilla RNN, LSTM, GRU) to predict
next-day binary market direction.

ANN Project — IEEE Format Report

---

## Overview

A modular pipeline that ingests financial price and news data, derives
sentiment signals (FinBERT + VADER for free-text, plus Alpha Vantage's
native Bullish/Neutral/Bearish labels), engineers a unified time-series
feature matrix, and trains/compares three sequential deep learning
architectures for next-day binary direction classification across five
large-cap US equities (AAPL, TSLA, NVDA, MSFT, AMZN).

---

## Project Structure

```
market-pulse-nn/
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb        ← Yahoo Finance, Alpha Vantage, NewsData.io, Reuters RSS
│   ├── 02_sentiment_analysis.ipynb    ← FinBERT scoring + VADER fallback (free-text sources)
│   ├── 03_feature_engineering.ipynb   ← Technical indicators + sentiment aggregation
│   ├── 04_model_training.ipynb        ← RNN / LSTM / GRU on Colab A100
│   └── 05_evaluation.ipynb            ← Metrics, curves, comparison report
│
├── src/
│   ├── ingestion/
│   │   ├── yahoo_finance.py            ← OHLCV downloader (yfinance)
│   │   ├── alpha_vantage.py            ← Alpha Vantage News Sentiment API client
│   │   ├── newsdata.py                 ← NewsData.io API client (cashtag queries)
│   │   └── rss_scraper.py              ← Reuters RSS parser (feedparser)
│   │
│   ├── sentiment/
│   │   └── analyzer.py                 ← FinBERT + VADER unified interface
│   │
│   ├── features/
│   │   ├── technical.py                ← RSI, MACD, Bollinger Bands, ATR
│   │   └── timeseries.py               ← Sliding window, scaler, train/val/test split
│   │
│   ├── models/
│   │   ├── rnn_model.py                ← Vanilla RNN (PyTorch nn.RNN)
│   │   ├── lstm_model.py               ← LSTM (PyTorch nn.LSTM)
│   │   ├── gru_model.py                ← GRU (PyTorch nn.GRU)
│   │   └── trainer.py                  ← Shared training loop, early stopping, checkpointing
│   │
│   └── utils/
│       ├── plotting.py                 ← All visualisation functions (curves, confusion matrix, etc.)
│       └── helpers.py                  ← Logging, seed fixing, data validation
│
├── data/
│   ├── raw/
│   │   ├── prices/                     ← OHLCV CSVs per ticker
│   │   ├── alphavantage/               ← Alpha Vantage news + sentiment scores (JSON)
│   │   ├── newsdata/                   ← NewsData.io articles (JSON)
│   │   └── rss/                        ← Reuters RSS articles (JSON)
│   ├── processed/                      ← Cleaned + sentiment-labelled data
│   └── final/                          ← Normalised sequences ready for model input
│
├── saved_models/                       ← .pt checkpoints (best val F1)
├── reports/
│   └── figures/                        ← All saved plots (PNG)
│
├── config.py                           ← Central config (paths, hyperparams, API keys)
├── .env.example                        ← API key template (copy → .env)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone & install (local — VS Code / macOS)

```bash
git clone https://github.com/<your-org>/market-pulse-nn.git
cd market-pulse-nn
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your API keys
```

### 2. Configure API keys

Edit `.env` with your Alpha Vantage and NewsData.io credentials.
Both providers offer free tiers sufficient for reproduction.

```env
ALPHAVANTAGE_API_KEY=your_key_here
NEWSDATA_API_KEY=your_key_here
```

Yahoo Finance and Reuters RSS require no authentication.

### 3. Run notebooks in order

| Notebook | Where to run | GPU needed |
|---|---|---|
| `01_data_ingestion.ipynb` | Local (VS Code) | No |
| `02_sentiment_analysis.ipynb` | Local or Colab | Recommended for FinBERT |
| `03_feature_engineering.ipynb` | Local (VS Code) | No |
| `04_model_training.ipynb` | **Colab A100** | Yes |
| `05_evaluation.ipynb` | Local or Colab | No |

### 4. Colab setup (for Notebooks 02 & 04)

```python
# Cell 1 — Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2 — Install deps
!pip install -r /content/drive/MyDrive/market-pulse-nn/requirements.txt -q

# Cell 3 — Add project to path
import sys
sys.path.insert(0, '/content/drive/MyDrive/market-pulse-nn')
```

`config.py` auto-detects the environment and resolves all paths to the
mounted Drive folder when running on Colab.

---

## Pipeline Summary

```
Data Ingestion → Sentiment Analysis → Feature Engineering
    → Sequence Construction → [RNN | LSTM | GRU] Training
        → Evaluation & Visualisation → IEEE Report
```

---

## Data Sources

| Source | Client | Data Type | Window |
|---|---|---|---|
| Yahoo Finance | `yfinance` | Daily OHLCV (5 tickers) | 2021-01-04 → 2024-12-31 |
| Alpha Vantage | REST | Pre-labelled news sentiment | 2021–2024 (sparse) |
| NewsData.io | REST | Free-text financial headlines | Recent (free-tier window) |
| Reuters RSS | `feedparser` | Free-text headlines | Polling-time onwards |

**Tickers tracked:** AAPL, TSLA, NVDA, MSFT, AMZN

---

## Sentiment Analysis

- **Alpha Vantage** articles arrive pre-scored with Bullish/Neutral/Bearish
  labels and a numeric `ticker_sentiment_score` in `[-1, +1]`.
- **NewsData.io and Reuters RSS** free-text is scored locally:
  - *Primary:* FinBERT (`ProsusAI/finbert`), batch size 64, max length 512.
  - *Fallback:* VADER (compound score threshold ±0.05).
- All per-document scores are aggregated into per-ticker daily features:
  mean sentiment, 3-day and 5-day rolling means, lag-1/2/3, and momentum.

---

## Models

| Model | Architecture | Parameters |
|---|---|---|
| Vanilla RNN | 2-layer, hidden=128, dropout=0.3 | ≈124K |
| LSTM | 2-layer, hidden=128, dropout=0.3 | ≈471K |
| GRU | 2-layer, hidden=128, dropout=0.3 | ≈355K |

All three share an identical training regime: AdamW optimiser (lr=1e-3,
wd=1e-5), BCE-with-logits loss, cosine-annealing scheduler, 100 epochs
with early stopping (patience=15, monitored on validation macro-F1),
and best-checkpoint saving by validation F1.

Lookback window: **60 trading days**, stride 1. Chronological 70/15/15
train/val/test split (no shuffling before split) to prevent look-ahead bias.

---

## Evaluation Metrics

- **Accuracy** — overall direction prediction correctness
- **F1-macro** — handles class imbalance
- **ROC-AUC** — threshold-independent discriminability
- **RMSE (probability)** — calibration of sigmoid outputs

---

## Results

Test set: **683 sequences** (chronologically held out, 55.3% UP days).

| Model | Accuracy | F1-macro | ROC-AUC | RMSE |
|---|---|---|---|---|
| Vanilla RNN | 0.4466 | 0.3087 | 0.5241 | 0.5019 |
| LSTM | 0.4466 | 0.3087 | 0.5176 | 0.5018 |
| **GRU** | **0.5417** | **0.5304** | **0.5237** | **0.4998** |

**Key finding:** Only GRU produced a non-degenerate decision boundary.
Vanilla RNN and LSTM both collapsed to predicting the Down class for every
test sample (hence identical accuracy and F1). All three models showed
near-identical ROC-AUC values just above the random baseline of 0.5,
consistent with the literature on equity direction prediction in
near-efficient markets.

See `reports/figures/` and the IEEE report for confusion matrices, ROC/PR
curves, and a full discussion of the model-collapse phenomenon and the
role of sparse sentiment coverage (only 6.5–10.8% of trading days carried
real sentiment signal under free-tier API constraints).

---

## Deliverables

- GitHub repository with this codebase
- IEEE-format Overleaf report covering Introduction, Methodology,
  Dataset Overview, Model Architecture Diagrams, Results Comparison,
  Challenges, Conclusion, and References
- All generated figures in `reports/figures/`
- Trained model checkpoints in `saved_models/`

---

## Authors

Group members: *Muhammad Musa*
Course: Artificial Neural Networks — FAST University CFD Campus