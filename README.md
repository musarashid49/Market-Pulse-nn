# market-pulse-nn

**Real-Time Market Movement Prediction using Sequential Deep Learning**  
ANN Project 1 — IEEE Format Report

---

## Overview

A production-quality pipeline that ingests live financial news and social media data,
derives sentiment signals, engineers time-series features, and trains three sequential
deep learning models (Vanilla RNN, LSTM, GRU) to predict next-day market direction,
price movement trends, and volatility spikes.

---

## Project Structure

```
market-pulse-nn/
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb        ← Yahoo Finance, Reuters RSS, Reddit, Twitter
│   ├── 02_sentiment_analysis.ipynb    ← FinBERT scoring + VADER fallback
│   ├── 03_feature_engineering.ipynb   ← Technical indicators + sentiment aggregation
│   ├── 04_model_training.ipynb        ← RNN / LSTM / GRU on Colab A100
│   └── 05_evaluation.ipynb            ← Metrics, curves, comparison report
│
├── src/
│   ├── ingestion/
│   │   ├── yahoo_finance.py            ← OHLCV downloader (yfinance)
│   │   ├── rss_scraper.py              ← Reuters RSS parser (feedparser)
│   │   ├── reddit_scraper.py           ← PRAW-based subreddit scraper
│   │   └── twitter_scraper.py          ← Tweepy v2 search client
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
│   │   ├── news/                       ← Reuters articles (JSON)
│   │   ├── reddit/                     ← Reddit posts + comments (JSON)
│   │   └── twitter/                    ← Tweets (JSON)
│   ├── processed/                      ← Cleaned + sentiment-labelled data
│   └── final/                          ← Normalised sequences ready for model input
│
├── saved_models/                       ← .pt checkpoints (best val F1)
├── reports/
│   └── figures/                        ← All saved plots (PNG, PDF)
│
├── config.py                           ← Central config (paths, hyperparams, API keys)
├── .env.example                        ← API key template (copy → .env)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone & install (local — VS Code / Mac)

```bash
git clone https://github.com/<your-org>/market-pulse-nn.git
cd market-pulse-nn
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your API keys
```

### 2. Configure API keys

Edit `.env` with your Reddit and Twitter credentials.  
See [`.env.example`](.env.example) for required keys.

### 3. Run notebooks in order

| Notebook | Where to run | GPU needed |
|---|---|---|
| `01_data_ingestion.ipynb` | Local (VS Code) | No |
| `02_sentiment_analysis.ipynb` | Local or Colab | Recommended |
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

---

## Pipeline Summary

```
Data Ingestion → Text Preprocessing → Sentiment Analysis
    → Feature Engineering → Sequence Construction
        → [RNN | LSTM | GRU] Training (parallel)
            → Evaluation & Metrics → Visualisation & Report
```

---

## Models

| Model | Architecture | Task |
|---|---|---|
| Vanilla RNN | 2-layer RNN, hidden=128, dropout=0.3 | Direction classification |
| LSTM | 2-layer LSTM, hidden=128, dropout=0.3 | Direction classification |
| GRU | 2-layer GRU, hidden=128, dropout=0.3 | Direction classification |

All models share the same training loop, scheduler (cosine annealing),
early stopping (patience=15), and are evaluated on identical test splits.

---

## Evaluation Metrics

- **Accuracy** — overall direction prediction correctness  
- **F1-score** — macro-averaged, handles class imbalance  
- **RMSE** — root mean square error on predicted probabilities (regression variant)  
- **ROC-AUC** — area under the receiver operating characteristic curve  

---

## Data Sources

| Source | Library | Data type |
|---|---|---|
| Yahoo Finance | `yfinance` | OHLCV price data |
| Reuters RSS | `feedparser` | Financial news headlines + body |
| Reddit | `praw` | Posts from r/wallstreetbets, r/stocks, r/investing |
| Twitter/X | `tweepy` | Tweets matching stock tickers |

---

## Deliverables

- GitHub repository with this codebase  
- Overleaf report (IEEE format) covering Introduction, Methodology, Dataset Overview,
  Model Architecture Diagrams, Results Comparison, Challenges, Conclusion, References  

---

## Authors

Group members: *[add your names here]*  
Course: Artificial Neural Networks  
