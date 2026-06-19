"""Fetch recent market-moving news via NewsAPI."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests


@dataclass
class NewsArticle:
    title: str
    description: Optional[str]
    source: str
    published_at: str


def fetch_market_news(tickers: list[str]) -> list[NewsArticle]:
    api_key = os.environ.get("NEWS_API_KEY", "")
    if not api_key:
        print("⚠️  NEWS_API_KEY not set — skipping news fetch")
        return []

    queries = [
        "stock market economy",
        "federal reserve inflation interest rates",
        "earnings revenue profit loss",
        *tickers[:3],
    ]

    seen: set[str] = set()
    articles: list[NewsArticle] = []

    for query in queries:
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 5,
                    "apiKey": api_key,
                },
                timeout=8,
            )
            resp.raise_for_status()
            for a in resp.json().get("articles", []):
                title = a.get("title", "") or ""
                if not title or title == "[Removed]" or title in seen:
                    continue
                seen.add(title)
                articles.append(NewsArticle(
                    title=title,
                    description=a.get("description"),
                    source=(a.get("source") or {}).get("name", "Unknown"),
                    published_at=a.get("publishedAt", ""),
                ))
        except Exception as exc:
            print(f"  News fetch failed for '{query}': {exc}")

        time.sleep(0.25)

    return articles
