"""Data ingestion sub-package."""
from .yahoo_finance   import YahooFinanceIngester
from .rss_scraper     import RSSNewsIngester
from .reddit_scraper  import RedditIngester
from .newsdata_scraper import NewsDataIngester

# TwitterIngester kept for backward compatibility but NewsDataIngester is recommended
from .twitter_scraper import TwitterIngester

__all__ = [
    "YahooFinanceIngester",
    "RSSNewsIngester",
    "RedditIngester",
    "NewsDataIngester",
    "TwitterIngester",
]
