"""
src/ingestion/yahoo_finance.py
==============================
OHLCV downloader wrapping yfinance.

Encapsulates all Yahoo Finance logic so notebooks stay clean and
the ingestion behaviour can be changed without editing notebooks.

Usage:
    from src.ingestion import YahooFinanceIngester
    ing = YahooFinanceIngester(tickers=CFG.TICKERS, save_dir=CFG.PRICES)
    data = ing.download(start=CFG.HIST_START, end=CFG.HIST_END)
    ing.save(data)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Optional
from ..utils.helpers import get_logger

logger = get_logger(__name__)


class YahooFinanceIngester:
    """
    Downloads, cleans, and persists daily OHLCV data for a list of tickers.

    Two derived columns are always added:
      log_return : ln(close_t / close_{t-1}) -- proportional daily move
      direction  : 1 if close_{t+1} > close_t, else 0  (TARGET LABEL)
    """

    def __init__(self, tickers: list, save_dir: Path) -> None:
        self.tickers  = tickers
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def download(
        self,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> dict:
        """
        Downloads OHLCV history for every ticker.

        Parameters
        ----------
        start    : start date string 'YYYY-MM-DD'
        end      : end date string   'YYYY-MM-DD'
        interval : yfinance interval -- '1d', '1h', etc.

        Returns
        -------
        dict[str, pd.DataFrame]  -- one cleaned DataFrame per ticker
        """
        results: dict = {}
        for ticker in self.tickers:
            logger.info(f"Downloading {ticker} ...")
            try:
                df = self._fetch_single(ticker, start, end, interval)
                if df is not None and not df.empty:
                    results[ticker] = df
                    logger.info(
                        f"  {ticker}: {len(df):,} rows | "
                        f"up {df['direction'].mean():.1%} | "
                        f"{df['date'].min().date()} -> {df['date'].max().date()}"
                    )
                else:
                    logger.warning(f"  {ticker}: no data returned.")
            except Exception as exc:
                logger.error(f"  {ticker}: download failed -- {exc}")
        return results

    # ------------------------------------------------------------------
    def _fetch_single(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        """Fetches, normalises, and annotates a single ticker."""
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=False,   # keep raw OHLC + adj_close separate
        )
        if raw.empty:
            return None

        # Flatten MultiIndex columns that yfinance >= 0.2 may return
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0].lower().replace(" ", "_") for col in raw.columns]
        else:
            raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

        # Standardise the adjusted-close column name
        for alt in ["adj close", "adj_close", "adjclose"]:
            if alt in raw.columns:
                raw = raw.rename(columns={alt: "adj_close"})
                break

        raw = raw.reset_index()
        raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
        raw["date"] = pd.to_datetime(raw.get("date", raw.get("datetime")))
        raw = raw.drop(columns=["datetime"], errors="ignore")
        raw["ticker"] = ticker

        # Log return: proportional, symmetric, ~normally distributed
        raw["log_return"] = np.log(raw["close"] / raw["close"].shift(1))

        # Direction label: shift -1 so today's row holds tomorrow's label
        raw["direction"] = (raw["close"].shift(-1) > raw["close"]).astype(int)

        # Drop the final row -- no tomorrow to label
        raw = raw.iloc[:-1].reset_index(drop=True)
        return raw

    # ------------------------------------------------------------------
    def save(self, data: dict) -> None:
        """Saves each ticker DataFrame as a CSV under save_dir."""
        for ticker, df in data.items():
            path = self.save_dir / f"{ticker}_daily.csv"
            df.to_csv(path, index=False)
            logger.info(f"Saved {path.name}")

    # ------------------------------------------------------------------
    def load(self, ticker: str) -> Optional[pd.DataFrame]:
        """Loads a previously saved ticker CSV. Returns None if missing."""
        path = self.save_dir / f"{ticker}_daily.csv"
        if not path.exists():
            logger.warning(f"{ticker}: file not found at {path}")
            return None
        return pd.read_csv(path, parse_dates=["date"])

    # ------------------------------------------------------------------
    def load_all(self) -> dict:
        """Loads every saved CSV in save_dir. Returns dict[ticker -> df]."""
        data = {}
        for csv_path in sorted(self.save_dir.glob("*_daily.csv")):
            ticker = csv_path.stem.replace("_daily", "")
            data[ticker] = pd.read_csv(csv_path, parse_dates=["date"])
        return data
