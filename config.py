"""
config.py — Central Configuration for market-pulse-nn
======================================================
Single source of truth for:
  - API credentials (loaded from .env)
  - Data paths (auto-switches between local VS Code and Colab)
  - Model hyperparameters
  - Feature engineering settings
  - Training settings

Usage:
    from config import CFG
    print(CFG.TICKERS)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load API keys from .env ───────────────────────────────────────────────────
load_dotenv()


# ═════════════════════════════════════════════════════════════════════════════
# Environment detection
# Automatically resolves paths whether running locally (VS Code/Mac)
# or on Google Colab (with Google Drive mounted at /content/drive).
# ═════════════════════════════════════════════════════════════════════════════

def _detect_env() -> str:
    """Returns 'colab' or 'local'."""
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        return "local"


ENV = _detect_env()

if ENV == "colab":
    PROJECT_ROOT = Path("/content/drive/MyDrive/market-pulse-nn")
else:
    PROJECT_ROOT = Path(__file__).resolve().parent


# ═════════════════════════════════════════════════════════════════════════════
# Paths
# ═════════════════════════════════════════════════════════════════════════════

class Paths:
    ROOT      = PROJECT_ROOT
    DATA_RAW  = ROOT / "data" / "raw"
    PRICES    = DATA_RAW / "prices"
    NEWS      = DATA_RAW / "news"
    REDDIT    = DATA_RAW / "reddit"
    TWITTER   = DATA_RAW / "twitter"
    DATA_PROC = ROOT / "data" / "processed"
    DATA_FINAL= ROOT / "data" / "final"
    MODELS    = ROOT / "saved_models"
    FIGURES   = ROOT / "reports" / "figures"


for _p in vars(Paths).values():
    if isinstance(_p, Path) and not _p.suffix:
        _p.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# Data Collection
# ═════════════════════════════════════════════════════════════════════════════

class DataConfig:
    TICKERS    = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]
    HIST_START = "2021-01-01"
    HIST_END   = "2024-12-31"
    YF_INTERVAL= "1d"

    # ── RSS feeds (working as of May 2026) ────────────────────────────────────
    RSS_FEEDS = [
        "https://finance.yahoo.com/news/rssindex",
        "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
        "https://www.investing.com/rss/news_25.rss",
        "https://www.marketwatch.com/rss/topstories",
    ]
    RSS_MAX_ARTICLES = 100

    # ── Reddit ────────────────────────────────────────────────────────────────
    REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID", "")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
    REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "market-pulse-nn/1.0")
    REDDIT_SUBREDDITS    = ["wallstreetbets", "stocks", "investing", "finance"]
    REDDIT_POST_LIMIT    = 500
    REDDIT_SORT          = "new"

    # ── Twitter/X (kept for compatibility, replaced by NewsData in practice) ──
    TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
    TWITTER_QUERY        = "(AAPL OR TSLA OR NVDA OR MSFT OR stocks) lang:en -is:retweet"
    TWITTER_MAX_RESULTS  = 100
    TWITTER_TWEET_FIELDS = ["created_at", "public_metrics", "entities"]


# ═════════════════════════════════════════════════════════════════════════════
# NewsData.io  (free: 2,000 articles/day — replaces Twitter)
# ═════════════════════════════════════════════════════════════════════════════

class NewsDataConfig:
    NEWSDATA_API_KEY      = os.getenv("NEWSDATA_API_KEY", "")
    NEWSDATA_LANGUAGE     = "en"
    NEWSDATA_CATEGORY     = "business"
    NEWSDATA_MAX_ARTICLES = 500
    NEWSDATA_SAVE_FILE    = "newsdata_articles.json"


# ═════════════════════════════════════════════════════════════════════════════
# Alpha Vantage  (free: 25 req/day — historical news sentiment 2021-2024)
# ═════════════════════════════════════════════════════════════════════════════

class AlphaVantageConfig:
    ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
    AV_NEWS_YEARS        = ["2021", "2022", "2023", "2024"]
    AV_MAX_PER_REQUEST   = 50     # free tier hard cap per request
    AV_SLEEP_BETWEEN     = 15     # seconds between requests


# ═════════════════════════════════════════════════════════════════════════════
# Sentiment Analysis
# ═════════════════════════════════════════════════════════════════════════════

class SentimentConfig:
    FINBERT_MODEL            = "ProsusAI/finbert"
    BATCH_SIZE               = 64
    MAX_LENGTH               = 512
    DEVICE                   = "cuda" if ENV == "colab" else "cpu"
    VADER_POSITIVE_THRESHOLD =  0.05
    VADER_NEGATIVE_THRESHOLD = -0.05
    SENTIMENT_ROLL_HOURS     = [1, 4, 24]


# ═════════════════════════════════════════════════════════════════════════════
# Feature Engineering
# ═════════════════════════════════════════════════════════════════════════════

class FeatureConfig:
    RSI_PERIOD       = 14
    MACD_FAST        = 12
    MACD_SLOW        = 26
    MACD_SIGNAL      = 9
    BB_PERIOD        = 20
    BB_STD           = 2
    ATR_PERIOD       = 14
    PRICE_LAGS       = [1, 2, 3, 5, 10]
    SENTIMENT_LAGS   = [1, 2, 3]
    TARGET           = "direction"
    FORECAST_HORIZON = 1


# ═════════════════════════════════════════════════════════════════════════════
# Sequence Construction
# ═════════════════════════════════════════════════════════════════════════════

class SeqConfig:
    LOOKBACK      = 60
    STRIDE        = 1
    TRAIN_RATIO   = 0.70
    VAL_RATIO     = 0.15
    TEST_RATIO    = 0.15
    RANDOM_SEED   = 42
    SHUFFLE_TRAIN = True


# ═════════════════════════════════════════════════════════════════════════════
# Model Hyperparameters  (shared across RNN / LSTM / GRU)
# ═════════════════════════════════════════════════════════════════════════════

class ModelConfig:
    NUM_LAYERS        = 2
    HIDDEN_SIZE       = 128
    DROPOUT           = 0.3
    BIDIRECTIONAL     = False
    EPOCHS            = 100
    BATCH_SIZE        = 64
    LEARNING_RATE     = 1e-3
    WEIGHT_DECAY      = 1e-5
    PATIENCE          = 15
    SCHEDULER         = "cosine"
    LOSS_FN           = "bce"
    SAVE_BEST         = True
    CHECKPOINT_METRIC = "val_f1"


# ═════════════════════════════════════════════════════════════════════════════
# CFG — single import for all notebooks
# e.g.  CFG.TICKERS | CFG.ALPHAVANTAGE_API_KEY | CFG.HIDDEN_SIZE
# ═════════════════════════════════════════════════════════════════════════════

class CFG(
    Paths,
    DataConfig,
    NewsDataConfig,
    AlphaVantageConfig,
    SentimentConfig,
    FeatureConfig,
    SeqConfig,
    ModelConfig,
):
    VERSION = "1.2.0"
    ENV     = ENV


if __name__ == "__main__":
    print(f"Environment          : {CFG.ENV}")
    print(f"Project root         : {CFG.ROOT}")
    print(f"Tickers              : {CFG.TICKERS}")
    print(f"Device               : {CFG.DEVICE}")
    print(f"Lookback             : {CFG.LOOKBACK} days")
    print(f"Hidden size          : {CFG.HIDDEN_SIZE}")
    print(f"Reddit key set       : {bool(CFG.REDDIT_CLIENT_ID)}")
    print(f"NewsData key set     : {bool(CFG.NEWSDATA_API_KEY)}")
    print(f"AlphaVantage key set : {bool(CFG.ALPHAVANTAGE_API_KEY)}")