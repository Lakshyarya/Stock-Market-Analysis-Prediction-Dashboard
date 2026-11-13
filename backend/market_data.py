"""Yahoo Finance adapter for global ticker discovery, price history and fundamentals.

Yahoo Finance coverage varies by exchange and is not an exchange-grade feed.
"""
import pandas as pd


def normalize_ticker(symbol, market='Global'):
    s = str(symbol or '').strip().upper().replace(' ', '')
    if not s:
        raise ValueError('Enter a company name or stock ticker.')
    # Preserve explicit exchange suffixes such as .NS, .L, .TO, .T, .HK, .DE.
    if '.' in s or market in ('US', 'Global'):
        return s
    if market == 'India':
        return s + '.NS'
    return s


def search_tickers(query, max_results=15):
    """Search Yahoo Finance listings by company name or ticker."""
    q = str(query or '').strip()
    if len(q) < 1:
        return []
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError('Install yfinance with: pip install yfinance') from exc

    try:
        result = yf.Search(q, max_results=max_results, news_count=0)
        quotes = getattr(result, 'quotes', None) or []
    except Exception as exc:
        raise RuntimeError(f'Global company search is unavailable: {exc}') from exc

    matches = []
    seen = set()
    for item in quotes:
        symbol = str(item.get('symbol') or '').strip().upper()
        quote_type = str(item.get('quoteType') or '').upper()
        if not symbol or symbol in seen:
            continue
        # Prioritize common shares; retain ETFs only when provider labels them as such.
        if quote_type and quote_type not in {'EQUITY', 'ETF'}:
            continue
        seen.add(symbol)
        name = item.get('shortname') or item.get('longname') or item.get('name') or symbol
        exchange = item.get('exchange') or item.get('exchDisp') or 'Exchange not supplied'
        matches.append({
            'symbol': symbol,
            'name': str(name),
            'exchange': str(exchange),
            'quoteType': quote_type.title() if quote_type else 'Security',
            'label': f"{name} ({symbol}) · {exchange}"
        })
    return matches


def fetch_history(symbol, period='2y', market='Global'):
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError('Install yfinance with: pip install yfinance') from exc
    ticker = normalize_ticker(symbol, market)
    hist = yf.download(ticker, period=period, interval='1d', auto_adjust=True, progress=False)
    if hist is None or hist.empty:
        raise ValueError(
            f'No historical data returned for {ticker}. Try selecting another exchange listing '
            'or entering the exact Yahoo Finance ticker with its exchange suffix.'
        )
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    hist = hist.reset_index()
    hist.columns = [str(c).title() for c in hist.columns]
    if 'Datetime' in hist and 'Date' not in hist:
        hist = hist.rename(columns={'Datetime':'Date'})
    if 'Adj Close' in hist and 'Close' not in hist:
        hist = hist.rename(columns={'Adj Close':'Close'})
    needed = ['Date','Open','High','Low','Close','Volume']
    missing = [c for c in needed if c not in hist]
    if missing:
        raise ValueError('Provider response missing: ' + ', '.join(missing))
    return hist[needed].dropna().reset_index(drop=True), ticker


def fetch_fundamentals(symbol, market='Global'):
    try:
        import yfinance as yf
        ticker = normalize_ticker(symbol, market)
        t = yf.Ticker(ticker)
        info = t.get_info() or {}
        financials = t.financials
        balance = t.balance_sheet
        cashflow = t.cashflow
        return info, financials, balance, cashflow
    except Exception:
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def fetch_quote_snapshot(symbols):
    """Return latest available provider quote snapshots for symbols.

    Tries recent intraday bars first, then daily bars. Yahoo Finance quotes may
    be delayed and are not an exchange-grade real-time feed.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError('Install yfinance with: pip install yfinance') from exc

    snapshots = {}
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        if not symbol:
            continue
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d', interval='1m', auto_adjust=True)
            intraday = hist is not None and not hist.empty and 'Close' in hist
            if not intraday:
                hist = ticker.history(period='5d', interval='1d', auto_adjust=True)
            if hist is None or hist.empty:
                snapshots[symbol] = {'available': False, 'error': 'No recent quote returned'}
                continue
            close = pd.to_numeric(hist['Close'], errors='coerce').dropna()
            if close.empty:
                snapshots[symbol] = {'available': False, 'error': 'No valid price'}
                continue
            latest = float(close.iloc[-1])
            # Use provider previous close when available; otherwise use the
            # preceding bar for a daily series or the prior intraday bar.
            previous = None
            try:
                previous = float(ticker.fast_info.get('previous_close'))
            except Exception:
                previous = None
            if previous is None or not pd.notna(previous) or previous <= 0:
                previous = float(close.iloc[-2]) if len(close) > 1 else latest
            change = latest - previous
            pct = (change / previous * 100) if previous else 0.0
            stamp = hist.index[-1]
            snapshots[symbol] = {
                'available': True, 'price': latest, 'previous_close': previous,
                'change': change, 'change_pct': pct, 'timestamp': str(stamp),
                'interval': '1m' if intraday else '1d',
            }
        except Exception as exc:
            snapshots[symbol] = {'available': False, 'error': str(exc)[:160]}
    return snapshots
