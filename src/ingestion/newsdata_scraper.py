"""
src/ingestion/newsdata_scraper.py
==================================
NewsData.io financial news scraper — free tier, no credit card required.

Free tier: 200 credits/day = 2,000 articles/day from 88,000+ sources
including Reuters, Bloomberg, CNBC, Financial Times, Seeking Alpha.

This replaces Twitter/X for this project because:
  - Twitter free tier = 1,500 tweets/month (too limited)
  - NewsData.io free tier = 2,000 articles/DAY (sufficient)
  - Financial news quality >> tweet quality for market prediction

Rate limits (free tier):
  - 30 credits per 15 minutes
  - We stay well within this with built-in sleep between requests

Usage:
    from src.ingestion.newsdata_scraper import NewsDataIngester
    scraper = NewsDataIngester(api_key=CFG.NEWSDATA_API_KEY, save_dir=CFG.TWITTER)
    df = scraper.fetch(tickers=CFG.TICKERS, max_articles=500)
    scraper.save(df)
"""

import time
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from ..utils.helpers import get_logger

logger = get_logger(__name__)

NEWSDATA_BASE_URL = "https://newsdata.io/api/1/latest"
SLEEP_BETWEEN_REQUESTS = 3.0   # seconds between API calls (stay within rate limit)


class NewsDataIngester:
    """
    Fetches recent financial news articles from NewsData.io for each ticker.

    Parameters
    ----------
    api_key  : NewsData.io API key (starts with 'pub_')
    save_dir : directory to write output JSON files
    """

    def __init__(self, api_key: str, save_dir: Path) -> None:
        self.api_key  = api_key
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def fetch(
        self,
        tickers: list,
        max_articles: int = 500,
        language: str = "en",
        category: str = "business",
    ) -> pd.DataFrame:
        """
        Fetches financial news articles mentioning each ticker symbol.

        Parameters
        ----------
        tickers      : list of ticker symbols, e.g. ['AAPL', 'TSLA']
        max_articles : total article budget across all tickers
        language     : article language code (default 'en')
        category     : NewsData.io category — 'business' covers financial news

        Returns
        -------
        pd.DataFrame with columns:
          article_id, title, description, content, source_id, source_name,
          published_at, link, ticker_ref, text
        """
        if not self.api_key or self.api_key == "pub_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX":
            logger.warning("NewsData.io API key not set — returning empty DataFrame.")
            logger.warning("Add NEWSDATA_API_KEY to .env and re-run.")
            return self._empty_df()

        per_ticker = max(10, max_articles // len(tickers))
        all_articles: list[dict] = []
        seen_ids: set[str] = set()

        for ticker in tickers:
            logger.info(f"Fetching news for {ticker} ...")
            articles = self._fetch_ticker(
                ticker=ticker,
                max_articles=per_ticker,
                language=language,
                category=category,
                seen_ids=seen_ids,
            )
            all_articles.extend(articles)
            logger.info(f"  {ticker}: {len(articles)} articles")
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        if not all_articles:
            logger.warning("No articles retrieved from NewsData.io.")
            return self._empty_df()

        df = pd.DataFrame(all_articles)
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        df = df.sort_values("published_at", ascending=False).reset_index(drop=True)

        logger.info(f"Total NewsData.io articles: {len(df):,}")
        return df

    # ------------------------------------------------------------------
    def _fetch_ticker(
        self,
        ticker: str,
        max_articles: int,
        language: str,
        category: str,
        seen_ids: set,
    ) -> list:
        """Fetches articles for a single ticker, handling pagination."""
        articles = []
        next_page = None

        while len(articles) < max_articles:
            params = {
                "apikey"  : self.api_key,
                "q"       : ticker,          # search for ticker symbol in article text
                "language": language,
                "category": category,
            }
            if next_page:
                params["page"] = next_page

            try:
                response = requests.get(NEWSDATA_BASE_URL, params=params, timeout=15)
                data     = response.json()
            except requests.exceptions.Timeout:
                logger.error(f"  {ticker}: request timed out.")
                break
            except Exception as exc:
                logger.error(f"  {ticker}: request failed — {exc}")
                break

            if response.status_code != 200 or data.get("status") != "success":
                self._log_api_error(ticker, response.status_code, data)
                break

            results = data.get("results", [])
            if not results:
                break

            for item in results:
                aid = item.get("article_id", "")
                if aid in seen_ids:
                    continue
                seen_ids.add(aid)

                title       = (item.get("title", "") or "").strip()
                description = (item.get("description", "") or "").strip()
                content     = (item.get("content", "") or "").strip()

                articles.append({
                    "article_id"  : aid,
                    "title"       : title,
                    "description" : description,
                    "content"     : content[:500],   # truncate for storage efficiency
                    "source_id"   : item.get("source_id", np.nan),
                    "source_name" : item.get("source_name", item.get("source_id", np.nan)),
                    "published_at": item.get("pubDate", np.nan),
                    "link"        : item.get("link", ""),
                    "ticker_ref"  : ticker,
                    # text = what gets sent to FinBERT in Notebook 02
                    "text"        : f"{title}. {description}".strip(),
                })

                if len(articles) >= max_articles:
                    break

            # Follow pagination if there are more results
            next_page = data.get("nextPage")
            if not next_page or len(articles) >= max_articles:
                break

            time.sleep(SLEEP_BETWEEN_REQUESTS)

        return articles

    # ------------------------------------------------------------------
    def save(
        self,
        df: pd.DataFrame,
        filename: str = "newsdata_articles.json",
    ) -> None:
        """Saves article DataFrame to JSON in save_dir."""
        if df.empty:
            logger.warning("No NewsData.io articles to save.")
            return
        export = df.copy()
        export["published_at"] = export["published_at"].apply(
            lambda ts: ts.isoformat() if pd.notna(ts) else None
        )
        path = self.save_dir / filename
        export.to_json(path, orient="records", indent=2, force_ascii=False)
        logger.info(f"Saved {path.name}  ({len(df):,} articles)")

    # ------------------------------------------------------------------
    def load(self, filename: str = "newsdata_articles.json") -> pd.DataFrame:
        """Loads a previously saved NewsData.io JSON file."""
        path = self.save_dir / filename
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return pd.DataFrame()
        df = pd.read_json(path, orient="records")
        if "published_at" in df.columns:
            df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def _empty_df() -> pd.DataFrame:
        """Returns an empty DataFrame with the correct schema."""
        return pd.DataFrame(columns=[
            "article_id", "title", "description", "content",
            "source_id", "source_name", "published_at", "link",
            "ticker_ref", "text",
        ])

    @staticmethod
    def _log_api_error(ticker: str, status_code: int, data: dict) -> None:
        """Parses and logs NewsData.io error responses."""
        if status_code == 401:
            logger.error(f"  {ticker}: 401 Unauthorized — check API key or verify your email at newsdata.io")
        elif status_code == 429:
            logger.warning(f"  {ticker}: 429 Rate limit reached — daily quota exhausted (resets at midnight UTC)")
        else:
            msg = data.get("results", {})
            if isinstance(msg, dict):
                msg = msg.get("message", str(data))
            logger.error(f"  {ticker}: HTTP {status_code} — {msg}")
