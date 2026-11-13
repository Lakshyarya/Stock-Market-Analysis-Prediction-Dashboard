"""Technical indicators and descriptive statistics for OHLCV data."""
import numpy as np
import pandas as pd


def add_indicators(df, ma_windows=(20, 50), vol_window=20):
    out = df.copy().sort_values('Date').reset_index(drop=True)
    close, high, low, volume = out['Close'], out['High'], out['Low'], out['Volume']
    out['Return'] = close.pct_change() * 100
    for n in sorted(set(list(ma_windows) + [20, 50, 200])):
        out[f'SMA{n}'] = close.rolling(n, min_periods=n).mean()
    out['EMA20'] = close.ewm(span=20, adjust=False).mean()
    out['EMA50'] = close.ewm(span=50, adjust=False).mean()
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out['RSI'] = (100 - 100 / (1 + rs)).fillna(50)
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    out['MACD'] = ema12 - ema26
    out['MACD_Signal'] = out['MACD'].ewm(span=9, adjust=False).mean()
    out['MACD_Hist'] = out['MACD'] - out['MACD_Signal']
    mid = close.rolling(20).mean(); sd = close.rolling(20).std()
    out['BB_Upper'], out['BB_Lower'] = mid + 2*sd, mid - 2*sd
    prev_close = close.shift(1)
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    out['ATR'] = tr.rolling(14).mean()
    out['Volatility'] = out['Return'].rolling(vol_window).std() * np.sqrt(252)
    out['Volume_SMA20'] = volume.rolling(20).mean()
    out['Drawdown'] = (close / close.cummax() - 1) * 100
    return out


def summary(df):
    r = df['Return'].dropna()
    return {
        'Start close': float(df['Close'].iloc[0]), 'End close': float(df['Close'].iloc[-1]),
        'Period return %': float((df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100),
        'Highest high': float(df['High'].max()), 'Lowest low': float(df['Low'].min()),
        'Avg volume': float(df['Volume'].mean()),
        'Daily volatility %': float(r.std()) if len(r) > 1 else float('nan'),
    }
