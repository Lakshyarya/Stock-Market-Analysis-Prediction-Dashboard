"""Loading, validating and cleaning historical OHLCV data."""
from pathlib import Path
import pandas as pd

REQUIRED = ["Date", "Open", "High", "Low", "Close", "Volume"]
NUMERIC = ["Open", "High", "Low", "Close", "Volume"]


def load_csv(source):
    """Read a CSV (path or file-like) and return (clean_df, quality_report)."""
    raw = pd.read_csv(source)
    raw.columns = [str(c).strip().title() for c in raw.columns]
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    return clean(raw[REQUIRED])


def clean(df):
    report = {"rows_loaded": len(df)}
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    report["missing_values"] = int(df.isna().sum().sum())
    df = df.dropna()

    bad = (
        (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
        | (df["Volume"] < 0)
        | (df["High"] < df["Low"])
        | (df["High"] < df[["Open", "Close"]].max(axis=1))
        | (df["Low"] > df[["Open", "Close"]].min(axis=1))
    )
    report["invalid_rows_removed"] = int(bad.sum())
    df = df[~bad]

    before = len(df)
    df = df.drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)
    report["duplicate_dates_removed"] = before - len(df)
    report["rows_after_cleaning"] = len(df)
    if df.empty:
        raise ValueError("No valid rows left after cleaning.")
    return df, report


def bundled_datasets(folder="data"):
    """Map stock name -> CSV path for every CSV in the data folder."""
    return {p.stem.upper(): p for p in sorted(Path(folder).glob("*.csv"))}
