# API Keys Setup Guide
## market-pulse-nn

Everything here is **free**. No credit card required for any service used in this project.

---

## Quick Summary

| Service | Purpose | Cost | Setup time |
|---|---|---|---|
| Yahoo Finance | OHLCV price data | Free, no key needed | 0 min |
| Reuters RSS | Financial news headlines | Free, no key needed | 0 min |
| Reddit | Social sentiment (r/wallstreetbets etc.) | Free | ~3 min |
| NewsData.io | Financial news from 88k+ sources | Free (2,000 articles/day) | ~2 min |

> Twitter/X is **not used** in this project. The free tier (1,500 tweets/month) is
> too limited to be useful. NewsData.io replaces it with higher-quality financial news.

---

## Step 1 — Yahoo Finance (nothing to do)

`yfinance` downloads OHLCV price data directly with no API key.
Run Notebook 01 and it just works.

---

## Step 2 — Reuters RSS (nothing to do)

Reuters publishes public RSS feeds. `feedparser` reads them without authentication.
Run Notebook 01 and it just works.

---

## Step 3 — Reddit API (3 minutes)

### 3.1 Create a Reddit account (skip if you have one)
Go to https://www.reddit.com and sign up.

### 3.2 Accept the Responsible Builder Policy
Go to: https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy
Click "I Agree" at the bottom. Required before creating any app.

### 3.3 Create your app
1. Go to: **https://www.reddit.com/prefs/apps**
2. Scroll to the bottom and click **"are you a developer? create an app..."**
3. Fill in the form exactly like this:

   ```
   name        : market-pulse-nn
   app type    : script          ← select this radio button
   description : ANN research project
   about url   : (leave blank)
   redirect uri: http://localhost:8080
   ```

4. Click **"create app"**

### 3.4 Copy your credentials
After creating the app, you will see a box like this:

```
market-pulse-nn
personal use script

[YOUR CLIENT ID IS HERE — 14 characters under the app name]

secret: [YOUR CLIENT SECRET IS HERE — 27 characters]
```

- The string **directly under "personal use script"** → `REDDIT_CLIENT_ID`
- The string next to **"secret:"** → `REDDIT_CLIENT_SECRET`
- Your Reddit username → fill in `REDDIT_USER_AGENT` below

### 3.5 Fill in .env
```
REDDIT_CLIENT_ID=AbCdEfGhIjKlMn          ← 14-char string under app name
REDDIT_CLIENT_SECRET=AbCdEfGhIjKlMnOpQrStUvWxYzA   ← 27-char secret
REDDIT_USER_AGENT=market-pulse-nn/1.0 by u/YourRedditUsername
```

---

## Step 4 — NewsData.io (2 minutes)

NewsData.io provides **2,000 financial news articles per day** free, from sources
including Reuters, Bloomberg, CNBC, Financial Times, and 88,000 others.

### 4.1 Register
1. Go to: **https://newsdata.io/register**
2. Fill in: name, email, password
3. Choose "Developer" as your role
4. Verify your email (check spam folder)

### 4.2 Get your API key
1. After login, go to: **https://newsdata.io/api-key**
   (or click "Dashboard" → "API Key" in the left sidebar)
2. Copy the key — it looks like: `pub_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`

### 4.3 Fill in .env
```
NEWSDATA_API_KEY=pub_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### Free tier limits
- 200 credits/day, 10 articles/credit = **2,000 articles/day**
- Rate limit: 30 credits per 15 minutes
- Articles delayed ~12 hours on free tier (fine for daily sentiment)
- Last 30 days of news (sufficient for live data; historical data in `data/raw/`)

---

## Step 5 — Validate everything

After filling in your `.env`, run this from your project root:

```bash
cd market-pulse-nn
python validate_keys.py
```

Expected output when everything is working:
```
[Yahoo Finance]  OK — downloaded 5 tickers
[Reuters RSS]    OK — 87 articles from 3 feeds
[Reddit]         OK — authenticated as u/YourUsername
[NewsData.io]    OK — 10 articles fetched (199 credits remaining today)

All 4 sources ready. Run notebooks in order: 01 -> 02 -> 03 -> 04 -> 05
```

---

## Troubleshooting

### Reddit: "401 Unauthorized"
- Double-check CLIENT_ID (it is under the app name, NOT your username)
- Make sure you accepted the Responsible Builder Policy first

### Reddit: "received 429 HTTP response"
- You are hitting the rate limit (100 req/min)
- PRAW handles this automatically — just wait a moment and retry

### NewsData.io: "You are not authorized"
- Your email may not be verified yet — check your inbox and spam

### NewsData.io: "rateLimitReached"
- You have used all 200 credits for today
- Credits reset at midnight UTC — re-run the next day
- For the project, one bulk pull is enough to build the dataset

### .env not loading
- Make sure `.env` is in the project root (same folder as `config.py`)
- Make sure there are no spaces around the `=` sign: `KEY=value` not `KEY = value`
- Make sure python-dotenv is installed: `pip install python-dotenv`
