"""
src/ingestion/rss_scraper.py
============================
Reuters / financial RSS feed parser using feedparser.

RSS is the cleanest free source of timestamped financial headlines:
  - No JavaScript rendering required
  - Structured XML with UTC timestamps
  - Designed for machine consumption -- legally clear

Usage:
    from src.ingestion import RSSNewsIngester
    scraper = RSSNewsIngester(feeds=CFG.RSS_FEEDS, save_dir=CFG.NEWS)
    df = scraper.fetch(max_per_feed=200)
    scraper.save(df)
"""

import re
import json
import numpy as np
import pandas as pd
import feedparser
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional
from ..utils.helpers import get_logger

logger = get_logger(__name__)


class RSSNewsIngester:
    """
    Parses a list of RSS feed URLs and returns a deduplicated DataFrame.

    Deduplication is by article URL so the same article appearing in
    multiple feeds is only stored once.
    """

    def __init__(self, feeds: list, save_dir: Path) -> None:
        self.feeds    = feeds
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def fetch(self, max_per_feed: int = 200) -> pd.DataFrame:
        """
        Fetches articles from all configured RSS feeds.

        Parameters
        ----------
        max_per_feed : maximum articles to retain per feed URL

        Returns
        -------
        pd.DataFrame sorted by published descending, columns:
          source, title, summary, published, link, author, text
        """
        all_articles: list[dict] = []
        seen_links: set[str]     = set()

        for feed_url in self.feeds:
            source_label = self._url_to_label(feed_url)
            logger.info(f"Fetching {feed_url} ...")
            parsed = feedparser.parse(feed_url)

            if parsed.bozo and not parsed.entries:
                logger.warning(
                    f"Bozo feed ({type(parsed.bozo_exception).__name__}) -- skipping."
                )
                continue

            count = 0
            for entry in parsed.entries[:max_per_feed]:
                link = entry.get("link", "")
                if link in seen_links:
                    continue
                seen_links.add(link)

                ts      = self._parse_timestamp(entry)
                title   = entry.get("title", "").strip()
                summary = self._clean_html(entry.get("summary", entry.get("description", "")))

                all_articles.append({
                    "source"   : source_label,
                    "title"    : title,
                    "summary"  : summary,
                    "published": ts,
                    "link"     : link,
                    "author"   : entry.get("author", np.nan),
                    # text = concatenated field sent to FinBERT
                    "text"     : f"{title}. {summary}".strip(),
                })
                count += 1

            logger.info(f"  {source_label}: {count} articles")

        if not all_articles:
            logger.warning("No articles retrieved from any feed.")
            return pd.DataFrame()

        df = pd.DataFrame(all_articles)
        df["published"] = pd.to_datetime(df["published"], utc=True)
        return df.sort_values("published", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    def save(self, df: pd.DataFrame, filename: str = "reuters_articles.json") -> None:
        """Saves DataFrame to JSON in save_dir."""
        if df.empty:
            logger.warning("Nothing to save -- DataFrame is empty.")
            return
        path   = self.save_dir / filename
        export = df.copy()
        export["published"] = export["published"].apply(
            lambda ts: ts.isoformat() if pd.notna(ts) else None
        )
        export.to_json(path, orient="records", indent=2, force_ascii=False)
        logger.info(f"Saved {path.name}  ({len(df):,} articles)")

    # ------------------------------------------------------------------
    def load(self, filename: str = "reuters_articles.json") -> pd.DataFrame:
        """Loads a previously saved news JSON file."""
        path = self.save_dir / filename
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return pd.DataFrame()
        df = pd.read_json(path, orient="records")
        df["published"] = pd.to_datetime(df["published"], utc=True)
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def _url_to_label(url: str) -> str:
        """Derives a short human-readable source label from the feed URL."""
        label = url.rstrip("/").split("/")[-1]
        label = label.replace("News", "").replace("news", "").lower()
        return f"reuters_{label}" if label else "reuters"

    @staticmethod
    def _parse_timestamp(entry) -> Optional[pd.Timestamp]:
        """Parses RFC 2822 or ISO 8601 timestamps robustly."""
        try:
            if getattr(entry, "published_parsed", None):
                return pd.Timestamp(
                    datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                )
            if entry.get("published"):
                return pd.Timestamp(parsedate_to_datetime(entry["published"]))
        except Exception:
            pass
        return pd.NaT

    @staticmethod
    def _clean_html(text: str) -> str:
        """Strips HTML tags and normalises whitespace."""
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
