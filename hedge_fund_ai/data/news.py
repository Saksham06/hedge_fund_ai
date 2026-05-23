from datetime import datetime

import requests

from hedge_fund_ai.config import NEWS_API_KEY


def fetch_news(ticker):

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": ticker,
        "pageSize": 5,
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY,
    }

    if not NEWS_API_KEY:
        return []

    try:
        r = requests.get(url, params=params, timeout=5)
        articles = r.json().get("articles", [])

        news = []
        for a in articles:
            if a.get("title") and a.get("publishedAt"):
                news.append(
                    {
                        "text": a["title"],
                        "published_at": datetime.fromisoformat(a["publishedAt"].replace("Z", "+00:00")),
                    }
                )

        return news

    except Exception:
        return []


def fetch_news_batch(tickers):

    news_map = {}

    for t in tickers:
        news_map[t] = fetch_news(t)

    return news_map
