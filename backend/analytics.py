"""Explainable heuristic stock scoring and risk metrics. Not investment advice."""
import numpy as np
import pandas as pd


def risk_metrics(df, benchmark=None):
    r = df['Close'].pct_change().dropna()
    if r.empty:
        return {}
    vol = r.std() * np.sqrt(252) * 100
    downside = r[r < 0]
    sharpe = (r.mean() / r.std() * np.sqrt(252)) if r.std() else np.nan
    peak = df['Close'].cummax()
    dd = (df['Close'] / peak - 1)
    var95 = np.percentile(r, 5) * 100
    beta = np.nan
    if benchmark is not None:
        aligned = pd.concat([r.rename('stock'), benchmark.pct_change().rename('market')], axis=1).dropna()
        if len(aligned) > 20 and aligned.market.var() > 0:
            beta = aligned.stock.cov(aligned.market) / aligned.market.var()
    return {'Annualized volatility':vol, 'Max drawdown':dd.min()*100,
            'Sharpe ratio (risk-free=0)':sharpe, 'Daily VaR 95%':var95,
            'Downside deviation (annualized)':downside.std()*np.sqrt(252)*100 if len(downside)>1 else np.nan,
            'Beta':beta}


def stock_score(df, info=None):
    info = info or {}
    latest = df.iloc[-1]
    # Each subscore is calculated only from available inputs; missing fundamentals are neutral, not fabricated.
    technical_parts = []
    if pd.notna(latest.get('RSI')):
        technical_parts.append(100 - min(abs(float(latest['RSI'])-55)*1.5, 70))
    if pd.notna(latest.get('SMA50')):
        technical_parts.append(75 if latest['Close'] > latest['SMA50'] else 35)
    if pd.notna(latest.get('MACD')):
        technical_parts.append(75 if latest['MACD'] > latest['MACD_Signal'] else 35)
    technical = float(np.mean(technical_parts)) if technical_parts else 50
    growth_parts=[]
    rev=info.get('revenueGrowth'); earn=info.get('earningsGrowth')
    for x in [rev,earn]:
        if isinstance(x,(int,float)) and np.isfinite(x): growth_parts.append(float(np.clip(50+x*100,0,100)))
    growth=float(np.mean(growth_parts)) if growth_parts else 50
    health_parts=[]
    roe=info.get('returnOnEquity'); margin=info.get('profitMargins'); debt=info.get('debtToEquity')
    for x in [roe,margin]:
        if isinstance(x,(int,float)) and np.isfinite(x): health_parts.append(float(np.clip(50+x*150,0,100)))
    if isinstance(debt,(int,float)) and np.isfinite(debt): health_parts.append(float(np.clip(100-debt/3,0,100)))
    health=float(np.mean(health_parts)) if health_parts else 50
    pe=info.get('trailingPE')
    valuation=float(np.clip(100-(pe-15)*1.5,10,90)) if isinstance(pe,(int,float)) and np.isfinite(pe) and pe>0 else 50
    vol=df['Close'].pct_change().std()*np.sqrt(252)*100
    risk=float(np.clip(100-vol*1.5,10,95)) if np.isfinite(vol) else 50
    components={'Financial health':health,'Growth':growth,'Technical strength':technical,'Valuation':valuation,'Risk profile':risk}
    weights={'Financial health':.30,'Growth':.20,'Technical strength':.20,'Valuation':.15,'Risk profile':.15}
    score=round(sum(components[k]*weights[k] for k in components))
    return score, components


def explain_signals(df):
    x=df.iloc[-1]; out=[]
    if pd.notna(x.get('RSI')):
        out.append(f"RSI is {x['RSI']:.1f}: {'overbought zone' if x['RSI']>=70 else 'oversold zone' if x['RSI']<=30 else 'between conventional 30/70 thresholds'}.")
    if pd.notna(x.get('SMA50')):
        out.append('Price is above its 50-day average.' if x['Close']>x['SMA50'] else 'Price is below its 50-day average.')
    if pd.notna(x.get('MACD')):
        out.append('MACD is above its signal line (positive momentum signal).' if x['MACD']>x['MACD_Signal'] else 'MACD is below its signal line (weaker momentum signal).')
    if pd.notna(x.get('Volatility')):
        out.append(f"Annualized historical volatility is approximately {x['Volatility']:.1f}%.")
    return out
