"""
src/ingestion/reddit_scraper.py
================================
PRAW-based Reddit post scraper for finance subreddits.

PRAW automatically handles OAuth2 and the 60 req/min rate limit.
Credentials must be set in .env (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET).

Usage:
    from src.ingestion import RedditIngester
    scraper = RedditIngester(
        client_id=CFG.REDDIT_CLIENT_ID,
        client_secret=CFG.REDDIT_CLIENT_SECRET,
        user_agent=CFG.REDDIT_USER_AGENT,
        save_dir=CFG.REDDIT,
    )
    df = scraper.scrape(subreddits=CFG.REDDIT_SUBREDDITS, limit=500)
    scraper.save(df)
"""

import re
import time
import pandas as pd
from pathlib import Path
from typing import Optional
from ..utils.helpers import get_logger

logger = get_logger(__name__)


class RedditIngester:
    """
    Scrapes posts from a list of finance subreddits using PRAW.

    Sort order:
      'new'  -- most recent regardless of score (best for time-series)
      'hot'  -- trending posts (biased toward viral events)
      'top'  -- top posts of all time (not useful for live sentiment)
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        save_dir: Path,
    ) -> None:
        self.client_id     = client_id
        self.client_secret = client_secret
        self.user_agent    = user_agent
        self.save_dir      = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._client       = None

    # ------------------------------------------------------------------
    def _get_client(self):
        """Lazy-initialises the PRAW client (avoids import at module load)."""
        if self._client is not None:
            return self._client

        import praw
        if not self.client_id or self.client_id == "your_reddit_client_id_here":
            raise ValueError(
                "Reddit credentials not set. "
                "Add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to .env"
            )

        self._client = praw.Reddit(
            client_id     = self.client_id,
            client_secret = self.client_secret,
            user_agent    = self.user_agent,
        )
        # Smoke test
        _ = list(self._client.subreddit("stocks").hot(limit=1))
        logger.info("Reddit client authenticated (read-only).")
        return self._client

    # ------------------------------------------------------------------
    def scrape(
        self,
        subreddits: list,
        limit: int = 500,
        sort: str = "new",
        sleep_between: float = 1.0,
    ) -> pd.DataFrame:
        """
        Scrapes posts from all configured subreddits.

        Parameters
        ----------
        subreddits     : list of subreddit names (without 'r/')
        limit          : max posts per subreddit (PRAW hard cap = 1000)
        sort           : 'new' | 'hot' | 'top'
        sleep_between  : seconds to pause between subreddits (server courtesy)

        Returns
        -------
        Combined pd.DataFrame with columns:
          post_id, subreddit, title, selftext, author, score,
          upvote_ratio, num_comments, created_utc, url, text
        """
        from praw.exceptions import PRAWException

        reddit = self._get_client()
        frames = []

        for sub_name in subreddits:
            logger.info(f"Scraping r/{sub_name} (sort={sort}, limit={limit}) ...")
            try:
                posts = self._scrape_one(reddit, sub_name, limit, sort)
                df    = pd.DataFrame(posts)
                frames.append(df)
                logger.info(
                    f"  r/{sub_name}: {len(df):,} posts | "
                    f"score median={int(df['score'].median()) if not df.empty else 'N/A'}"
                )
                time.sleep(sleep_between)
            except PRAWException as exc:
                logger.error(f"  r/{sub_name}: PRAW error -- {exc}")
            except Exception as exc:
                logger.error(f"  r/{sub_name}: unexpected error -- {exc}")

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        logger.info(f"Total Reddit posts: {len(combined):,}")
        return combined

    # ------------------------------------------------------------------
    @staticmethod
    def _scrape_one(reddit, sub_name: str, limit: int, sort: str) -> list:
        """Collects posts from a single subreddit."""
        sub    = reddit.subreddit(sub_name)
        getter = {"new": sub.new, "hot": sub.hot, "top": sub.top}.get(sort, sub.new)
        posts  = []

        for submission in getter(limit=limit):
            # Skip deleted / removed posts
            if submission.selftext in ("[deleted]", "[removed]"):
                continue

            title_clean = re.sub(r"\s+", " ", submission.title or "").strip()
            body_clean  = re.sub(r"\s+", " ", submission.selftext or "").strip()

            posts.append({
                "post_id"      : submission.id,
                "subreddit"    : sub_name,
                "title"        : title_clean,
                "selftext"     : body_clean,
                "author"       : str(submission.author) if submission.author else "[deleted]",
                "score"        : submission.score,
                "upvote_ratio" : submission.upvote_ratio,
                "num_comments" : submission.num_comments,
                "created_utc"  : pd.Timestamp(submission.created_utc, unit="s", tz="UTC"),
                "url"          : submission.url,
                "text"         : f"{title_clean} {body_clean}".strip(),
            })

        return posts

    # ------------------------------------------------------------------
    def save(self, df: pd.DataFrame, filename: str = "all_subreddits_combined.json") -> None:
        """Saves combined DataFrame to JSON. Also saves one file per subreddit."""
        if df.empty:
            logger.warning("Nothing to save.")
            return

        # Per-subreddit files
        for sub_name in df["subreddit"].unique():
            sub_df  = df[df["subreddit"] == sub_name].copy()
            sub_df  = self._serialise_timestamps(sub_df)
            sub_path = self.save_dir / f"{sub_name}_posts.json"
            sub_df.to_json(sub_path, orient="records", indent=2, force_ascii=False)
            logger.info(f"Saved {sub_path.name}  ({len(sub_df):,} posts)")

        # Combined file
        combined = self._serialise_timestamps(df.copy())
        path = self.save_dir / filename
        combined.to_json(path, orient="records", indent=2, force_ascii=False)
        logger.info(f"Saved combined: {path.name}  ({len(df):,} posts)")

    # ------------------------------------------------------------------
    def load(self, filename: str = "all_subreddits_combined.json") -> pd.DataFrame:
        """Loads a previously saved Reddit JSON file."""
        path = self.save_dir / filename
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return pd.DataFrame()
        df = pd.read_json(path, orient="records")
        df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def _serialise_timestamps(df: pd.DataFrame) -> pd.DataFrame:
        """Converts Timestamp columns to ISO strings for JSON export."""
        for col in ["created_utc"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda ts: ts.isoformat() if pd.notna(ts) else None
                )
        return df
