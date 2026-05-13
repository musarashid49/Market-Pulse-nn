"""
src/ingestion/twitter_scraper.py
================================
Tweepy v2 cashtag search client for stock-related tweets.

Uses the Twitter v2 Recent Search endpoint (last 7 days on Basic tier).
Cashtag search ($AAPL vs AAPL) is more precise -- only intentional
investment discussion uses cashtags.

Rate limits (Basic tier):
  1 request / 15 seconds, 100 tweets / request, 10,000 tweets / month

Usage:
    from src.ingestion import TwitterIngester
    scraper = TwitterIngester(bearer_token=CFG.TWITTER_BEARER_TOKEN, save_dir=CFG.TWITTER)
    df = scraper.search(tickers=CFG.TICKERS, max_per_ticker=100)
    scraper.save(df)
"""

import time
import pandas as pd
from pathlib import Path
from typing import Optional
from ..utils.helpers import get_logger

logger = get_logger(__name__)

# Conservative budget cap -- adjust based on your API tier
DEFAULT_MAX_PER_TICKER = 100


class TwitterIngester:
    """
    Fetches recent tweets mentioning stock tickers via Tweepy v2.

    Graceful degradation: if the bearer token is missing or quota is
    exceeded, methods return an empty DataFrame instead of raising.
    """

    def __init__(self, bearer_token: str, save_dir: Path) -> None:
        self.bearer_token = bearer_token
        self.save_dir     = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._client      = None

    # ------------------------------------------------------------------
    def _get_client(self):
        """Lazy-initialises the Tweepy client."""
        if self._client is not None:
            return self._client

        import tweepy

        if not self.bearer_token or self.bearer_token == "your_twitter_bearer_token_here":
            raise ValueError(
                "Twitter Bearer Token not set. "
                "Add TWITTER_BEARER_TOKEN to .env"
            )
        self._client = tweepy.Client(
            bearer_token=self.bearer_token,
            wait_on_rate_limit=True,   # auto-sleep on 429s
        )
        logger.info("Tweepy v2 client ready.")
        return self._client

    # ------------------------------------------------------------------
    def search(
        self,
        tickers: list,
        max_per_ticker: int = DEFAULT_MAX_PER_TICKER,
        sleep_between: float = 15.0,
    ) -> pd.DataFrame:
        """
        Searches recent tweets for each ticker's cashtag.

        Parameters
        ----------
        tickers         : list of ticker symbols, e.g. ['AAPL', 'TSLA']
        max_per_ticker  : max tweets per ticker (API cap = 100 per request)
        sleep_between   : seconds between requests (Basic: 1 req / 15 sec)

        Returns
        -------
        pd.DataFrame with columns:
          tweet_id, text, author_id, created_at,
          like_count, retweet_count, reply_count, query, ticker_ref
        """
        import tweepy

        try:
            client = self._get_client()
        except ValueError as exc:
            logger.warning(str(exc))
            return pd.DataFrame()

        all_tweets: list[dict] = []

        for ticker in tickers:
            # Cashtag + English only + exclude retweets (keep original opinions)
            query = f"${ticker} lang:en -is:retweet"
            logger.info(f"Searching ${ticker} ...")

            try:
                response = client.search_recent_tweets(
                    query=query,
                    max_results=min(max_per_ticker, 100),
                    tweet_fields=["created_at", "public_metrics", "author_id"],
                )
            except tweepy.errors.TweepyException as exc:
                logger.error(f"  API error for ${ticker}: {exc}")
                time.sleep(sleep_between)
                continue

            if not response.data:
                logger.info(f"  ${ticker}: no results.")
                time.sleep(sleep_between)
                continue

            for tweet in response.data:
                m = tweet.public_metrics or {}
                all_tweets.append({
                    "tweet_id"      : str(tweet.id),
                    "text"          : tweet.text,
                    "author_id"     : str(tweet.author_id),
                    "created_at"    : pd.Timestamp(tweet.created_at, tz="UTC"),
                    "like_count"    : m.get("like_count",    0),
                    "retweet_count" : m.get("retweet_count", 0),
                    "reply_count"   : m.get("reply_count",   0),
                    "query"         : query,
                    "ticker_ref"    : ticker,
                })

            logger.info(f"  ${ticker}: {len(response.data)} tweets")
            time.sleep(sleep_between)

        if not all_tweets:
            return pd.DataFrame(columns=[
                "tweet_id", "text", "author_id", "created_at",
                "like_count", "retweet_count", "reply_count", "query", "ticker_ref",
            ])

        df = pd.DataFrame(all_tweets)
        logger.info(f"Total tweets collected: {len(df):,}")
        return df

    # ------------------------------------------------------------------
    def save(self, df: pd.DataFrame, filename: str = "tweets.json") -> None:
        """Saves tweet DataFrame to JSON."""
        if df.empty:
            logger.warning("No tweet data to save.")
            return
        export = df.copy()
        export["created_at"] = export["created_at"].apply(
            lambda ts: ts.isoformat() if pd.notna(ts) else None
        )
        path = self.save_dir / filename
        export.to_json(path, orient="records", indent=2, force_ascii=False)
        logger.info(f"Saved {path.name}  ({len(df):,} tweets)")

    # ------------------------------------------------------------------
    def load(self, filename: str = "tweets.json") -> pd.DataFrame:
        """Loads a previously saved tweets JSON file."""
        path = self.save_dir / filename
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return pd.DataFrame()
        df = pd.read_json(path, orient="records")
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        return df
