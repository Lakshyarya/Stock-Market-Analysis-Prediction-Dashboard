"""Leakage-aware next-session direction classification and model evaluation.

The model estimates the probability that the next available daily close is above
or below the latest close. It does not estimate a target price.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["Return", "Prev_Return", "Close_vs_MA", "Volume_Change", "Price_Change"]
MODELS = {
    "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=250, max_depth=6, min_samples_leaf=3, class_weight="balanced_subsample", random_state=42),
}


def _feature_frame(df, ma_window=20):
    d = pd.DataFrame(index=df.index)
    d["Return"] = df["Close"].pct_change() * 100
    d["Prev_Return"] = d["Return"].shift(1)
    d["Close_vs_MA"] = df["Close"] / df["Close"].rolling(ma_window).mean() - 1
    d["Volume_Change"] = df["Volume"].pct_change().replace([float("inf"), -float("inf")], pd.NA) * 100
    d["Price_Change"] = (df["Close"] - df["Open"]) / df["Open"].replace(0, pd.NA) * 100
    return d.replace([float("inf"), -float("inf")], pd.NA)


def build_dataset(df, ma_window=20):
    """Features use data through day t; label indicates whether day t+1 rises."""
    d = _feature_frame(df, ma_window)
    d["Label"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    if len(d):
        d.loc[d.index[-1], "Label"] = pd.NA
    return d.dropna(subset=FEATURES + ["Label"])


def run(df, model_name="Random Forest", test_size=0.2, ma_window=20):
    if model_name not in MODELS:
        raise ValueError(f"Unsupported model. Choose one of: {', '.join(MODELS)}")
    ordered = df.sort_values("Date").reset_index(drop=True).copy()
    if len(ordered) < ma_window + 60:
        raise ValueError(f"Need at least {ma_window + 60} daily observations; received {len(ordered)}.")

    data = build_dataset(ordered, ma_window)
    if len(data) < 60:
        raise ValueError("Not enough usable rows after feature preparation. Check Open, Close and Volume columns for missing or zero values.")

    split = int(len(data) * (1 - test_size))
    train, test = data.iloc[:split], data.iloc[split:]
    if train["Label"].nunique() < 2:
        raise ValueError("Training history contains only one direction class; a classifier cannot be trained reliably.")
    if test.empty:
        raise ValueError("Not enough test observations for a chronological evaluation.")

    evaluation_model = MODELS[model_name]().fit(train[FEATURES], train["Label"].astype(int))
    y_test = test["Label"].astype(int)
    predictions = evaluation_model.predict(test[FEATURES])
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average="binary", zero_division=0)
    baseline = max(y_test.mean(), 1 - y_test.mean())

    # Refit on every labeled observation, then score the latest row whose next-day
    # outcome is not yet known. This is a genuine forward-looking inference step.
    final_model = MODELS[model_name]().fit(data[FEATURES], data["Label"].astype(int))
    latest_features = _feature_frame(ordered, ma_window).dropna(subset=FEATURES)
    if latest_features.empty or latest_features.index[-1] != ordered.index[-1]:
        raise ValueError("Could not build prediction features for the latest trading session. Check the latest OHLCV values.")
    latest_x = latest_features[FEATURES].iloc[[-1]]
    probabilities = final_model.predict_proba(latest_x)[0]
    class_prob = {int(cls): float(prob) for cls, prob in zip(final_model.classes_, probabilities)}
    up_probability = class_prob.get(1, 0.0)
    down_probability = class_prob.get(0, 0.0)
    direction = "Up" if up_probability >= down_probability else "Down"

    if model_name == "Random Forest":
        importance = pd.Series(evaluation_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    else:
        importance = pd.Series(evaluation_model[-1].coef_[0], index=FEATURES).abs().sort_values(ascending=False)

    return {
        "accuracy": accuracy_score(y_test, predictions), "precision": precision, "recall": recall, "f1": f1,
        "baseline": baseline, "train_rows": len(train), "test_rows": len(test),
        "confusion": confusion_matrix(y_test, predictions, labels=[0, 1]), "importance": importance,
        "prediction": {
            "direction": direction,
            "probability_up": up_probability,
            "probability_down": down_probability,
            "confidence": max(up_probability, down_probability),
            "as_of": pd.Timestamp(ordered["Date"].iloc[-1]).isoformat(),
            "horizon": "Next trading session",
            "latest_close": float(ordered["Close"].iloc[-1]),
        },
    }

# Direct multi-horizon return forecasting. Targets are future cumulative returns
# from the observation date to the requested trading-session horizon.
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

FORECAST_HORIZONS = {
    "week": {"label": "1 week", "sessions": 5},
    "month": {"label": "1 month", "sessions": 21},
    "6month": {"label": "6 months", "sessions": 126},
    "year": {"label": "1 year", "sessions": 252},
}
FORECAST_FEATURES = [
    "ret_1", "ret_5", "ret_21", "ret_63", "vol_21", "vol_63",
    "price_sma20", "price_sma50", "price_sma200", "volume_change",
    "range_pct", "momentum_126",
]


def _forecast_features(df):
    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")
    open_price = pd.to_numeric(df["Open"], errors="coerce")
    r = close.pct_change()
    x = pd.DataFrame(index=df.index)
    x["ret_1"] = r
    x["ret_5"] = close.pct_change(5)
    x["ret_21"] = close.pct_change(21)
    x["ret_63"] = close.pct_change(63)
    x["vol_21"] = r.rolling(21).std() * np.sqrt(252)
    x["vol_63"] = r.rolling(63).std() * np.sqrt(252)
    for window in (20, 50, 200):
        x[f"price_sma{window}"] = close / close.rolling(window).mean() - 1
    x["volume_change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan).clip(-5, 5)
    x["range_pct"] = (pd.to_numeric(df["High"], errors="coerce") - pd.to_numeric(df["Low"], errors="coerce")) / open_price.replace(0, np.nan)
    x["momentum_126"] = close.pct_change(126)
    return x.replace([np.inf, -np.inf], np.nan)


def forecast_horizon(df, horizon_key):
    """Train/evaluate a direct Random Forest return model for one horizon."""
    if horizon_key not in FORECAST_HORIZONS:
        raise ValueError("Unsupported forecast horizon.")
    spec = FORECAST_HORIZONS[horizon_key]
    h = spec["sessions"]
    ordered = df.sort_values("Date").reset_index(drop=True).copy()
    if len(ordered) < max(320, h + 100):
        raise ValueError(f"Not enough daily history for a {spec['label']} forecast. Need at least {max(320, h + 100)} observations; received {len(ordered)}.")

    features = _forecast_features(ordered)
    close = pd.to_numeric(ordered["Close"], errors="coerce")
    target = close.shift(-h) / close - 1
    dataset = features.copy()
    dataset["target"] = target
    dataset = dataset.dropna(subset=FORECAST_FEATURES + ["target"])
    if len(dataset) < 100:
        raise ValueError("Not enough valid observations after feature engineering for this forecast horizon.")

    # Purge the final h training rows so their target windows do not overlap test dates.
    split = int(len(dataset) * 0.8)
    train_end = max(0, split - h)
    train = dataset.iloc[:train_end]
    test = dataset.iloc[split:]
    if len(train) < 60 or len(test) < 20:
        raise ValueError(f"Insufficient independent train/test observations for the {spec['label']} horizon. Try a longer history or a shorter horizon.")

    params = dict(n_estimators=300, max_depth=5, min_samples_leaf=8, max_features=0.8, random_state=42, n_jobs=-1)
    evaluation_model = RandomForestRegressor(**params).fit(train[FORECAST_FEATURES], train["target"])
    test_pred = evaluation_model.predict(test[FORECAST_FEATURES])
    actual = test["target"].to_numpy()
    mae = float(mean_absolute_error(actual, test_pred))
    rmse = float(np.sqrt(mean_squared_error(actual, test_pred)))
    directional = float(np.mean((test_pred >= 0) == (actual >= 0)))
    baseline_directional = float(np.mean((np.zeros_like(actual) >= 0) == (actual >= 0)))

    final_model = RandomForestRegressor(**params).fit(dataset[FORECAST_FEATURES], dataset["target"])
    latest = features.dropna(subset=FORECAST_FEATURES).iloc[[-1]]
    if latest.empty or latest.index[-1] != ordered.index[-1]:
        raise ValueError("Latest market row does not contain enough valid features for a forecast.")
    tree_forecasts = np.array([tree.predict(latest[FORECAST_FEATURES].to_numpy())[0] for tree in final_model.estimators_])
    predicted_return = float(final_model.predict(latest[FORECAST_FEATURES])[0])
    latest_close = float(close.iloc[-1])
    predicted_price = latest_close * (1 + predicted_return)
    return {
        "horizon": horizon_key,
        "horizon_label": spec["label"],
        "trading_sessions": h,
        "as_of": pd.to_datetime(ordered["Date"].iloc[-1]).date().isoformat(),
        "latest_close": latest_close,
        "predicted_return_pct": predicted_return * 100,
        "predicted_price": float(predicted_price),
        "direction": "Bullish" if predicted_return > 0.005 else "Bearish" if predicted_return < -0.005 else "Sideways",
        "model": "Random Forest Regressor",
        "evaluation": {
            "mae_return_pct": mae * 100,
            "rmse_return_pct": rmse * 100,
            "directional_accuracy_pct": directional * 100,
            "baseline_directional_accuracy_pct": baseline_directional * 100,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        },
        "tree_prediction_spread_pct": float(np.std(tree_forecasts) * 100),
    }
