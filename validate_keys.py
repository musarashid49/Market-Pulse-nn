"""
validate_keys.py
================
Run this script AFTER filling in your .env to confirm every data source
is reachable and authenticated before running any notebooks.

Usage (from the market-pulse-nn project root):
    python validate_keys.py

Expected output when everything is working:
    [Yahoo Finance]  OK  — 5/5 tickers downloaded
    [Reuters RSS]    OK  — 87 articles from 3 feeds
    [Reddit]         OK  — authenticated as u/YourUsername
    [NewsData.io]    OK  — 10 articles fetched | 199 credits remaining today

    All 4 sources ready.  Run notebooks in order: 01 -> 02 -> 03 -> 04 -> 05
"""

import sys
import os
from pathlib import Path

# ── Make sure the project root is on the path ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

results = {}   # source -> (ok: bool, message: str)

# ── Colour helpers (works on Mac terminal) ────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
OK     = f"{GREEN}  OK  {RESET}"
FAIL   = f"{RED}  FAIL{RESET}"
SKIP   = f"{YELLOW}  SKIP{RESET}"


# =============================================================================
# 1. Yahoo Finance
# =============================================================================
def check_yahoo_finance():
    print("[Yahoo Finance]  Testing ...", end=" ", flush=True)
    try:
        import yfinance as yf
        tickers = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]
        ok_count = 0
        for t in tickers:
            df = yf.download(t, period="5d", progress=False, auto_adjust=False)
            if not df.empty:
                ok_count += 1
        if ok_count == len(tickers):
            return True, f"{ok_count}/{len(tickers)} tickers downloaded"
        else:
            return False, f"Only {ok_count}/{len(tickers)} tickers returned data"
    except ImportError:
        return False, "yfinance not installed — run: pip install yfinance"
    except Exception as e:
        return False, f"Error: {e}"


# =============================================================================
# 2. Reuters RSS
# =============================================================================
def check_reuters_rss():
    print("[Reuters RSS]    Testing ...", end=" ", flush=True)
    try:
        import feedparser
        feeds = [
            "https://finance.yahoo.com/news/rssindex",
            "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
            "https://www.investing.com/rss/news_25.rss",
            "https://www.marketwatch.com/rss/topstories",
        ]
        total = 0
        for url in feeds:
            parsed = feedparser.parse(url)
            total += len(parsed.entries)
        if total > 0:
            return True, f"{total} articles from {len(feeds)} feeds"
        else:
            return False, "Feeds returned 0 articles (may be a network issue)"
    except ImportError:
        return False, "feedparser not installed — run: pip install feedparser"
    except Exception as e:
        return False, f"Error: {e}"


# =============================================================================
# 3. Reddit
# =============================================================================
def check_reddit():
    print("[Reddit]         Testing ...", end=" ", flush=True)

    client_id     = os.getenv("REDDIT_CLIENT_ID", "").strip()
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
    user_agent    = os.getenv("REDDIT_USER_AGENT", "market-pulse-nn/1.0").strip()

    if not client_id or not client_secret:
        return None, "Credentials not set in .env (see API_KEYS_SETUP.md Step 3)"

    if "YOUR_REDDIT_USERNAME" in user_agent:
        return False, "Replace YOUR_REDDIT_USERNAME in REDDIT_USER_AGENT"

    try:
        import praw
        reddit = praw.Reddit(
            client_id     = client_id,
            client_secret = client_secret,
            user_agent    = user_agent,
        )
        # Lightweight smoke test: fetch 1 hot post from r/stocks
        posts = list(reddit.subreddit("stocks").hot(limit=1))
        if posts:
            username = reddit.user.me()   # returns None for read-only scripts without username/password
            user_str = f"u/{username}" if username else "read-only mode"
            return True, f"Authenticated | {user_str} | r/stocks accessible"
        else:
            return False, "Connected but r/stocks returned no posts"
    except ImportError:
        return False, "praw not installed — run: pip install praw"
    except Exception as e:
        err = str(e)
        if "401" in err:
            return False, "401 Unauthorized — check CLIENT_ID and CLIENT_SECRET"
        if "403" in err:
            return False, "403 Forbidden — accept the Responsible Builder Policy first"
        return False, f"Error: {err}"


# =============================================================================
# 4. NewsData.io
# =============================================================================
def check_newsdata():
    print("[NewsData.io]    Testing ...", end=" ", flush=True)

    api_key = os.getenv("NEWSDATA_API_KEY", "").strip()

    if not api_key:
        return None, "API key not set in .env (see API_KEYS_SETUP.md Step 4)"

    try:
        import requests
        url    = "https://newsdata.io/api/1/latest"
        params = {
            "apikey"  : api_key,
            "q"       : "stock market",
            "language": "en",
            "category": "business",
        }
        r = requests.get(url, params=params, timeout=15)
        data = r.json()

        if r.status_code == 200 and data.get("status") == "success":
            articles = data.get("results", [])
            # Some endpoints return credit info in the response
            credits_left = data.get("creditLeft", "N/A")
            return True, f"{len(articles)} articles fetched | {credits_left} credits remaining today"

        # Handle common errors
        err_code = data.get("results", {})
        if isinstance(err_code, dict):
            msg = err_code.get("message", r.text)
        else:
            msg = data.get("message", r.text)

        if r.status_code == 401:
            return False, f"401 Unauthorized — key may be wrong or email not verified"
        if r.status_code == 429:
            return False, "429 Rate limit — you have used all 200 credits today (resets at midnight UTC)"
        return False, f"HTTP {r.status_code}: {msg}"

    except ImportError:
        return False, "requests not installed — run: pip install requests"
    except requests.exceptions.Timeout:
        return False, "Request timed out — check internet connection"
    except Exception as e:
        return False, f"Error: {e}"


# =============================================================================
# Run all checks and print the report
# =============================================================================

print()
print("=" * 60)
print("  market-pulse-nn — API Key Validation")
print("=" * 60)
print()

checks = [
    ("Yahoo Finance",  check_yahoo_finance),
    ("Reuters RSS",    check_reuters_rss),
    ("Reddit",         check_reddit),
    ("NewsData.io",    check_newsdata),
]

all_ok      = True
any_skipped = False

for name, fn in checks:
    ok, msg = fn()
    if ok is True:
        status = OK
        results[name] = True
    elif ok is None:
        status = SKIP
        results[name] = None
        any_skipped   = True
    else:
        status = FAIL
        results[name] = False
        all_ok         = False
    print(f"  {status}  {msg}")

print()
print("─" * 60)
print()

if all_ok and not any_skipped:
    print(f"{GREEN}  All 4 sources ready.{RESET}")
    print("  Run notebooks in order:  01 -> 02 -> 03 -> 04 -> 05")
elif any_skipped:
    skipped = [k for k, v in results.items() if v is None]
    failed  = [k for k, v in results.items() if v is False]
    if not failed:
        print(f"{YELLOW}  Sources not yet configured:{RESET} {', '.join(skipped)}")
        print("  Follow API_KEYS_SETUP.md to add them, then re-run this script.")
        print()
        print("  The project works with just Yahoo Finance + Reuters RSS.")
        print("  Reddit and NewsData.io add social sentiment signal (recommended).")
    else:
        print(f"{RED}  Fix the FAIL items above before running the notebooks.{RESET}")
        print("  See API_KEYS_SETUP.md for step-by-step instructions.")
else:
    print(f"{RED}  Fix the FAIL items before running the notebooks.{RESET}")
    print("  See API_KEYS_SETUP.md for step-by-step help.")

print()
