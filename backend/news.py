"""Marketaux news adapter and transparent event-reaction helpers."""
from __future__ import annotations
import os
import requests
import pandas as pd

BASE_URL = "https://api.marketaux.com/v1/news/all"


def clean_symbol(symbol: str) -> str:
    """Convert Yahoo-style tickers to provider-neutral symbols (e.g. RELIANCE.NS -> RELIANCE)."""
    return str(symbol or "").strip().upper().replace(".NS", "").replace(".BO", "")


def fetch_company_news(symbol: str, limit: int = 15, api_key: str | None = None) -> list[dict]:
    token = api_key or os.getenv("MARKETAUX_API_KEY", "")
    if not token:
        raise RuntimeError("Market news is not configured. Add MARKETAUX_API_KEY to your .env file.")
    params = {
        "api_token": token,
        "symbols": clean_symbol(symbol),
        "filter_entities": "true",
        "language": "en",
        "limit": max(1, min(int(limit), 50)),
    }
    response = requests.get(BASE_URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    articles = payload.get("data", []) or []
    normalized = []
    for item in articles:
        entities = item.get("entities") or []
        matching = [e for e in entities if clean_symbol(e.get("symbol", "")) == clean_symbol(symbol)]
        sentiment = None
        if matching:
            sentiment = matching[0].get("sentiment_score")
        if sentiment is None:
            scores = [e.get("sentiment_score") for e in entities if isinstance(e.get("sentiment_score"), (int, float))]
            sentiment = sum(scores) / len(scores) if scores else None
        try:
            sentiment = float(sentiment) if sentiment is not None else None
        except (TypeError, ValueError):
            sentiment = None
        normalized.append({
            "title": item.get("title") or "Untitled article",
            "description": item.get("description") or item.get("snippet") or "No article summary supplied.",
            "url": item.get("url", ""),
            "source": item.get("source", "Unknown source"),
            "published_at": item.get("published_at", ""),
            "sentiment_score": sentiment,
        })
    return normalized


def sentiment_label(score):
    if score is None or pd.isna(score):
        return "Unrated"
    if score >= 0.15:
        return "Positive"
    if score <= -0.15:
        return "Negative"
    return "Neutral"


def event_reaction(article: dict, history: pd.DataFrame) -> dict:
    """Estimate close-to-close move around publication; this is association, not causation."""
    if history is None or history.empty or not article.get("published_at"):
        return {"reaction_pct": None, "window": "Unavailable"}
    try:
        published = pd.to_datetime(article["published_at"], utc=True).tz_convert(None).normalize()
        dates = pd.to_datetime(history["Date"]).dt.tz_localize(None).dt.normalize()
        before = history.loc[dates < published]
        after = history.loc[dates >= published]
        if before.empty or after.empty:
            return {"reaction_pct": None, "window": "Insufficient price history"}
        p0 = float(before.iloc[-1]["Close"])
        p1 = float(after.iloc[0]["Close"])
        if p0 <= 0:
            return {"reaction_pct": None, "window": "Unavailable"}
        return {"reaction_pct": (p1 / p0 - 1) * 100, "window": f"{before.iloc[-1]['Date'].date()} close → {after.iloc[0]['Date'].date()} close"}
    except Exception:
        return {"reaction_pct": None, "window": "Unavailable"}
