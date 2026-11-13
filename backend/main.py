from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import math
import os
import sys

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import market_data
import indicators
import analytics
import ml
import news as news_service

try:
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
except ImportError:
    pass


app = FastAPI(title="Stock Sense API", version="2.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

POPULAR = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "JPM", "LLY", "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "7203.T",
]
INDICES = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW JONES": "^DJI",
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
}


def clean_symbol(symbol: str) -> str:
    return market_data.normalize_ticker(symbol, "Global")


def json_safe(value):
    """Recursively convert pandas/NumPy values to strict-JSON-safe values.

    JSON does not support NaN or +/-Infinity. These are represented as null.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())

    if isinstance(value, (pd.Series,)):
        return json_safe(value.to_dict())

    if isinstance(value, (pd.DataFrame,)):
        return json_safe(value.to_dict(orient="records"))

    if isinstance(value, (pd.Timestamp, datetime, date)):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, np.generic):
        return json_safe(value.item())

    if isinstance(value, (float,)):
        return value if math.isfinite(value) else None

    if isinstance(value, (int, str, bool)):
        return value

    # Handle scalar pandas NA/NaN values without applying pd.isna to containers.
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def json_value(value):
    return json_safe(value)


def finite_float(value, default=None):
    """Return a finite Python float, otherwise default."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def records(df: pd.DataFrame, limit: int | None = None):
    """Convert a DataFrame to JSON-safe records, preserving nulls for NaNs."""
    if df is None or df.empty:
        return []

    out = df.copy()
    if limit is not None:
        out = out.tail(limit)

    if "Date" in out.columns:
        dates = pd.to_datetime(out["Date"], errors="coerce")
        out["Date"] = dates.dt.strftime("%Y-%m-%d")

    # Cast to object before replacing missing values; otherwise float columns
    # can retain IEEE NaN even after .where(..., None).
    out = out.astype(object)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.where(pd.notna(out), None)

    return json_safe(out.to_dict(orient="records"))


def metric_dict(values):
    """Convert a metric mapping to finite floats or nulls."""
    return {str(k): finite_float(v) for k, v in dict(values or {}).items()}


def get_history(symbol: str, period: str = "1y"):
    try:
        hist, resolved = market_data.fetch_history(symbol, period, "Global")
        hist, quality = __import__("data").clean(hist)

        if hist is None or not isinstance(hist, pd.DataFrame) or hist.empty:
            raise ValueError(f"No usable historical observations were returned for {symbol}.")

        required = {"Date", "Close"}
        missing = required.difference(hist.columns)
        if missing:
            raise ValueError("Processed history is missing required columns: " + ", ".join(sorted(missing)))

        hist = hist.copy()
        hist["Close"] = pd.to_numeric(hist["Close"], errors="coerce")
        hist = hist.loc[hist["Close"].notna() & np.isfinite(hist["Close"])].reset_index(drop=True)

        if hist.empty:
            raise ValueError(f"No valid closing prices were returned for {symbol}.")

        return hist, resolved, json_safe(quality)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "Stock Sense API",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/search")
def search(q: str = Query(min_length=1, max_length=80)):
    try:
        return json_safe({"results": market_data.search_tickers(q, 15)})
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/markets")
def markets():
    try:
        snap = market_data.fetch_quote_snapshot(list(INDICES.values()))
        result = []
        for name, ticker in INDICES.items():
            row = snap.get(ticker, {})
            result.append({"name": name, "symbol": ticker, **row})
        return json_safe({
            "data": result,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": "Yahoo Finance; may be delayed",
        })
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/quotes")
def quotes(symbols: str = ",".join(POPULAR)):
    requested = [s.strip().upper() for s in symbols.split(",") if s.strip()][:30]
    try:
        snap = market_data.fetch_quote_snapshot(requested)
        data = []
        for symbol in requested:
            item = snap.get(symbol, {"available": False, "error": "No quote returned"})
            data.append({"symbol": symbol, **item})
        return json_safe({
            "data": data,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": "Yahoo Finance; delayed/availability varies",
        })
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/stock/{symbol}")
def stock_overview(symbol: str, period: str = "1y"):
    hist, resolved, quality = get_history(symbol, period)
    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No historical data available for {resolved}.")

    try:
        enriched = indicators.add_indicators(hist)
        if enriched is None or enriched.empty:
            raise ValueError("Technical indicators could not be calculated for this stock.")

        info, _, _, _ = market_data.fetch_fundamentals(resolved, "Global")
        info = info or {}
        score, components = analytics.stock_score(enriched, info)
        risk = analytics.risk_metrics(enriched)

        last = enriched.iloc[-1]
        prev = enriched.iloc[-2] if len(enriched) > 1 else last
        last_close = finite_float(last.get("Close"))
        prev_close = finite_float(prev.get("Close"))
        if last_close is None:
            raise ValueError("The latest closing price is unavailable.")

        change = (last_close - prev_close) if prev_close is not None else 0.0
        change_pct = ((last_close / prev_close - 1) * 100) if prev_close not in (None, 0) else 0.0

        payload = {
            "symbol": resolved,
            "name": info.get("longName") or info.get("shortName") or resolved,
            "exchange": info.get("exchange") or info.get("fullExchangeName") or "Exchange data unavailable",
            "currency": json_value(info.get("currency")) or "USD",
            "sector": json_value(info.get("sector")),
            "industry": json_value(info.get("industry")),
            "website": json_value(info.get("website")),
            "market_cap": json_value(info.get("marketCap")),
            "pe": json_value(info.get("trailingPE")),
            "price": last_close,
            "change": change,
            "change_pct": change_pct,
            "high_52w": json_value(info.get("fiftyTwoWeekHigh")),
            "low_52w": json_value(info.get("fiftyTwoWeekLow")),
            "volume": finite_float(last.get("Volume")),
            "score": json_safe(score),
            "score_components": json_safe(components),
            "risk": metric_dict(risk),
            "signals": json_safe(analytics.explain_signals(enriched)),
            "source": "Yahoo Finance · latest available historical quote",
            "latest_observation": json_safe(pd.to_datetime(last.get("Date"), errors="coerce")),
            "quality": quality,
            "history": records(enriched, 300),
        }
        return json_safe(payload)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Unable to build stock analysis for {resolved}: {str(exc)}") from exc


@app.get("/api/stock/{symbol}/fundamentals")
def fundamentals(symbol: str):
    try:
        info, fin, balance, cashflow = market_data.fetch_fundamentals(clean_symbol(symbol), "Global")
        info = info or {}

        def frame_payload(frame):
            if frame is None or frame.empty:
                return []
            x = frame.copy()
            x.columns = [
                str(pd.to_datetime(c, errors="coerce").date())
                if not isinstance(c, str) and not pd.isna(pd.to_datetime(c, errors="coerce"))
                else str(c)
                for c in x.columns
            ]
            x = x.astype(object).replace([np.inf, -np.inf], np.nan)
            x = x.where(pd.notna(x), None)
            return json_safe([
                {"metric": str(idx), **{str(col): json_value(val) for col, val in row.items()}}
                for idx, row in x.iterrows()
            ])

        selected = {
            k: json_value(info.get(k))
            for k in [
                "longName", "sector", "industry", "marketCap", "trailingPE", "forwardPE",
                "priceToBook", "returnOnEquity", "returnOnAssets", "profitMargins",
                "operatingMargins", "revenueGrowth", "earningsGrowth", "debtToEquity",
                "dividendYield", "trailingEps", "totalRevenue", "grossMargins",
            ]
        }
        return json_safe({
            "info": selected,
            "income_statement": frame_payload(fin),
            "balance_sheet": frame_payload(balance),
            "cash_flow": frame_payload(cashflow),
        })
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Fundamentals unavailable: {str(exc)}") from exc


@app.get("/api/stock/{symbol}/news")
def company_news(symbol: str, limit: int = Query(default=15, ge=1, le=50)):
    try:
        articles = news_service.fetch_company_news(symbol, limit=limit)
        for article in articles:
            article["sentiment"] = news_service.sentiment_label(article.get("sentiment_score"))
        return json_safe({
            "data": articles,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/stock/{symbol}/ml")
def stock_ml(symbol: str, period: str = "5y", model: str = "Random Forest"):
    hist, resolved, _ = get_history(symbol, period)
    enriched = indicators.add_indicators(hist)
    try:
        result = ml.run(enriched, model)
        importance = result["importance"]
        payload = {
            "symbol": resolved,
            "model": model,
            "accuracy": finite_float(result.get("accuracy")),
            "precision": finite_float(result.get("precision")),
            "recall": finite_float(result.get("recall")),
            "f1": finite_float(result.get("f1")),
            "baseline": finite_float(result.get("baseline")),
            "train_rows": int(result["train_rows"]),
            "test_rows": int(result["test_rows"]),
            "prediction": result["prediction"],
            "importance": [
                {"feature": str(k), "importance": finite_float(v)}
                for k, v in importance.items()
            ],
        }
        return json_safe(payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/stock/{symbol}/forecast")
def stock_forecast(symbol: str, horizon: str = Query(default="week", pattern="^(week|month|6month|year)$")):
    """Forecast the selected stock over a week, month, six months or year."""
    hist, resolved, _ = get_history(symbol, "10y")
    try:
        result = ml.forecast_horizon(hist, horizon)
        return json_safe({"symbol": resolved, "source": "Yahoo Finance historical OHLCV", **result})
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Forecast unavailable for {resolved}: {str(exc)}") from exc


@app.get("/api/stock/{symbol}/risk")
def stock_risk(symbol: str, period: str = "2y"):
    hist, resolved, _ = get_history(symbol, period)
    try:
        payload = {
            "symbol": resolved,
            "metrics": metric_dict(analytics.risk_metrics(hist)),
            "history": records(hist, 500),
        }
        return json_safe(payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Risk metrics unavailable: {str(exc)}") from exc




@app.post("/api/datasets/analyze")
async def analyze_uploaded_dataset(
    file: UploadFile = File(...),
    mode: str = Form("single"),
):
    """Analyze an uploaded historical-price CSV/XLSX without persisting the file."""
    name = (file.filename or "").lower()
    if not (name.endswith(".csv") or name.endswith(".xlsx")):
        raise HTTPException(status_code=415, detail="Upload a CSV or XLSX file.")
    raw = await file.read(12 * 1024 * 1024 + 1)
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds the 12 MB upload limit.")
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    try:
        from io import BytesIO
        if name.endswith(".csv"):
            frame = pd.read_csv(BytesIO(raw))
        else:
            frame = pd.read_excel(BytesIO(raw), engine="openpyxl")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read this file. Check that it is a valid CSV or XLSX.")
    if frame.empty or len(frame) < 3:
        raise HTTPException(status_code=422, detail="Dataset must contain at least 3 data rows.")
    if len(frame) > 250000:
        raise HTTPException(status_code=413, detail="Dataset exceeds the 250,000-row analysis limit.")
    frame.columns = [str(c).strip() for c in frame.columns]
    if len(frame.columns) > 100:
        raise HTTPException(status_code=422, detail="Dataset has too many columns (maximum 100).")
    lower = {str(c).strip().lower(): c for c in frame.columns}
    def col(*names):
        return next((lower[n] for n in names if n in lower), None)
    date_col = col("date", "datetime", "timestamp", "time")
    if date_col is None:
        raise HTTPException(status_code=422, detail="A Date column is required.")
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce", utc=True)
    frame = frame.dropna(subset=[date_col]).copy()
    frame[date_col] = frame[date_col].dt.tz_convert(None)
    if mode not in ("single", "portfolio"):
        raise HTTPException(status_code=422, detail="Mode must be single or portfolio.")

    close_col = col("close", "adj close", "adjusted close", "price")
    symbol_col = col("symbol", "ticker", "asset")
    warnings = []
    if mode == "single":
        if close_col is None:
            raise HTTPException(status_code=422, detail="Single-stock mode requires a Close (or Price) column.")
        frame[close_col] = pd.to_numeric(frame[close_col], errors="coerce")
        frame = frame.dropna(subset=[close_col])
        frame = frame[frame[close_col] > 0].sort_values(date_col).drop_duplicates(date_col, keep="last")
        if len(frame) < 3:
            raise HTTPException(status_code=422, detail="At least 3 valid positive closing prices are required.")
        hist = pd.DataFrame({"Close": frame[close_col].astype(float).to_numpy()},
                            index=pd.DatetimeIndex(frame[date_col]))
        metrics = analytics.risk_metrics(hist)
        ret = hist["Close"].pct_change().dropna()
        peak = hist["Close"].cummax()
        drawdown = (hist["Close"] / peak - 1) * 100
        if len(frame) < 30: warnings.append("Fewer than 30 observations: risk estimates may be unstable.")
        result = {
            "mode": "single", "asset_count": 1, "rows": int(len(hist)),
            "date_start": hist.index.min().date().isoformat(),
            "date_end": hist.index.max().date().isoformat(),
            "metrics": {**metric_dict(metrics), "Daily CVaR 95%": float(ret[ret <= ret.quantile(.05)].mean()*100) if len(ret) else None},
            "statistics": {"mean_daily_return": float(ret.mean()*100),
                           "best_day": float(ret.max()*100), "worst_day": float(ret.min()*100),
                           "daily_return_std": float(ret.std()*100),
                           "historical_cvar_95": float(ret[ret <= ret.quantile(.05)].mean()*100) if len(ret) else None},
            "history": [{"Date": d.strftime("%Y-%m-%d"), "Close": float(v),
                         "Drawdown": float(dd)} for d,v,dd in zip(hist.index, hist.Close, drawdown)],
            "warnings": warnings,
            "assumptions": ["Annualization uses 252 trading days.", "Sharpe ratio assumes a 0% risk-free rate.",
                            "Daily VaR 95% is the 5th percentile of historical daily returns.",
                            "Beta is unavailable unless benchmark data is supplied."]
        }
    else:
        # Accept either long format (Date, Symbol, Close[, Weight]) or wide format (Date, one column per asset).
        if symbol_col is not None and close_col is not None:
            frame[close_col] = pd.to_numeric(frame[close_col], errors="coerce")
            frame = frame.dropna(subset=[symbol_col, close_col])
            frame = frame[frame[close_col] > 0]
            wide = frame.pivot_table(index=date_col, columns=symbol_col, values=close_col, aggfunc="last")
            weight_col = col("weight", "allocation", "portfolio weight")
            weights_raw = {}
            if weight_col:
                for sym, group in frame.groupby(symbol_col):
                    vals = pd.to_numeric(group[weight_col], errors="coerce").dropna()
                    if len(vals): weights_raw[str(sym)] = float(vals.iloc[-1])
        else:
            value_cols = [c for c in frame.columns if c != date_col]
            for c in value_cols: frame[c] = pd.to_numeric(frame[c], errors="coerce")
            wide = frame.set_index(date_col)[value_cols]
            wide = wide.loc[:, wide.notna().sum() >= 3]
            weights_raw = {}
        wide = wide.sort_index().replace([np.inf, -np.inf], np.nan)
        wide = wide.dropna(axis=1, thresh=3)
        if wide.shape[1] < 2:
            raise HTTPException(status_code=422, detail="Portfolio mode needs at least two assets. Use Date/Symbol/Close rows or Date plus multiple price columns.")
        wide = wide.ffill().dropna()
        wide = wide[wide.gt(0).all(axis=1)]
        if len(wide) < 3:
            raise HTTPException(status_code=422, detail="Not enough overlapping valid price dates across the assets.")
        if weights_raw and all(str(c) in weights_raw for c in wide.columns):
            weights = np.array([weights_raw[str(c)] for c in wide.columns], dtype=float)
            if np.any(weights < 0) or weights.sum() <= 0:
                raise HTTPException(status_code=422, detail="Portfolio weights must be non-negative and have a positive total.")
            weights = weights / weights.sum()
            weight_note = "User-provided weights normalized to 100%."
        else:
            weights = np.repeat(1/len(wide.columns), len(wide.columns))
            weight_note = "Equal weights assumed because no complete Weight/Allocation column was provided."
        returns = wide.pct_change().dropna()
        port_returns = returns.dot(weights)
        port_value = (1 + port_returns).cumprod()
        peak = port_value.cummax()
        dd = (port_value / peak - 1) * 100
        vol = float(port_returns.std()*np.sqrt(252)*100)
        var = float(np.percentile(port_returns, 5)*100)
        cvar = float(port_returns[port_returns <= np.percentile(port_returns,5)].mean()*100)
        sharpe = float(port_returns.mean()/port_returns.std()*np.sqrt(252)) if port_returns.std() else None
        cov = returns.cov().to_numpy() * 252
        contributions = (weights * (cov @ weights)) / max(float(weights @ cov @ weights), 1e-12)
        corr = returns.corr()
        result = {
            "mode":"portfolio", "asset_count":int(wide.shape[1]), "rows":int(len(wide)),
            "date_start":wide.index.min().date().isoformat(), "date_end":wide.index.max().date().isoformat(),
            "metrics":{"Annualized volatility":vol, "Max drawdown":float(dd.min()),
                       "Daily VaR 95%":var, "Daily CVaR 95%":cvar, "Sharpe ratio (risk-free=0)":sharpe},
            "weights":[{"symbol":str(s),"weight":float(w),"risk_contribution":float(rc*100)}
                       for s,w,rc in zip(wide.columns,weights,contributions)],
            "correlation":{"symbols":[str(x) for x in corr.columns],
                           "matrix":corr.replace([np.inf,-np.inf],np.nan).fillna(0).round(4).values.tolist()},
            "history":[{"Date":d.strftime("%Y-%m-%d"),"Value":float(v),"Drawdown":float(x)}
                       for d,v,x in zip(port_value.index,port_value,dd)],
            "warnings":warnings, "assumptions":[weight_note,
                "Returns are aligned on common dates after forward-filling missing prices.",
                "Weights are treated as buy-and-hold; no periodic rebalancing or transaction costs are modeled.",
                "Annualization uses 252 trading days."]
        }
    return json_safe(result)


class AskRequest(BaseModel):
    question: str


@app.post("/api/stock/{symbol}/ask")
def ask_analyst(symbol: str, body: AskRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Enter a question first.")

    hist, resolved, _ = get_history(symbol, "1y")
    enriched = indicators.add_indicators(hist)
    info, _, _, _ = market_data.fetch_fundamentals(resolved, "Global")
    info = info or {}
    score, components = analytics.stock_score(enriched, info)
    risk = metric_dict(analytics.risk_metrics(enriched))

    latest = enriched.iloc[-1]
    latest_close = finite_float(latest.get("Close"))
    if latest_close is None:
        raise HTTPException(status_code=422, detail="Latest closing price is unavailable.")

    snapshot = json_safe({
        "symbol": resolved,
        "latest_close": latest_close,
        "latest_observation": pd.to_datetime(latest.get("Date"), errors="coerce"),
        "score": score,
        "score_components": components,
        "risk": risk,
        "signals": analytics.explain_signals(enriched),
        "fundamentals": {
            k: json_value(info.get(k))
            for k in [
                "sector", "industry", "trailingPE", "returnOnEquity",
                "profitMargins", "revenueGrowth", "debtToEquity",
            ]
        },
    })

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        component_text = ", ".join(
            f"{k}: {finite_float(v, 0):.0f}/100"
            for k, v in (components or {}).items()
        )
        return json_safe({
            "answer": (
                "AI is not configured. Here is the deterministic analysis available "
                "without an AI key: "
                + " ".join(snapshot["signals"])
                + f" Stock Sense score: {score}/100. Components: {component_text}. "
                "Add OPENAI_API_KEY to backend/.env to enable question-specific AI explanations."
            ),
            "ai_enabled": False,
            "snapshot": snapshot,
        })

    try:
        from openai import OpenAI

        response = OpenAI(api_key=api_key).responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are Stock Sense, a cautious financial research explainer. "
                        "Use only the supplied snapshot facts. Clearly distinguish observed "
                        "data from interpretation. Never invent missing numbers or claim "
                        "certainty. Explain jargon simply. Do not give personalized investment "
                        "advice or guaranteed predictions."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Analysis snapshot: {snapshot}\nQuestion: {question}",
                },
            ],
        )
        return json_safe({
            "answer": response.output_text,
            "ai_enabled": True,
            "snapshot": snapshot,
        })
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI request failed: {str(exc)[:240]}") from exc


@app.get("/api/watchlist")
def watchlist():
    try:
        snap = market_data.fetch_quote_snapshot(POPULAR[:10])
        return json_safe({
            "data": [
                {"symbol": s, **snap.get(s, {"available": False})}
                for s in POPULAR[:10]
            ]
        })
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
