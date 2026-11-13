# Stock Sense — multi-horizon stock forecasts

This build adds an ML stock forecast card to the Overview page for the currently selected company. Select any supported company using global search or a company row, then choose **1 Week**, **1 Month**, **6 Months**, or **1 Year** and click **Predict stock**.

## What the forecast does

- Fetches up to 10 years of daily, auto-adjusted OHLCV history from Yahoo Finance.
- Builds lagged returns, momentum, rolling volatility, price-to-moving-average, volume-change and daily-range features.
- Trains a separate Random Forest regressor for the selected horizon (5, 21, 126 or 252 trading sessions).
- Uses chronological train/test evaluation and purges overlapping horizon labels at the split.
- Displays an estimated future price, estimated percentage change, direction, holdout directional accuracy, baseline directional accuracy, MAE and test sample count.

The prediction is a model estimate, not a guaranteed price target or investment recommendation. Results can be weak or unavailable for recently listed companies, thinly traded securities, or symbols with insufficient provider history. Yahoo Finance coverage and delay vary by exchange.

## Run locally (Windows)

Backend terminal:

```cmd
cd backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Frontend terminal:

```cmd
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The Vite development proxy forwards `/api` requests to `http://127.0.0.1:8000`.

## API

`GET /api/stock/{symbol}/forecast?horizon=week|month|6month|year`

Example: `/api/stock/NVDA/forecast?horizon=month`

## Notes

- This feature predicts a horizon-end price estimate and direction. It is not an intraday trading signal.
- A forecast for each horizon is trained independently; the 6-month and 1-year forecasts are not compounded from the 1-week forecast.
- The displayed directional accuracy is a historical holdout metric, not the probability that the current forecast will be correct.
- The tree prediction spread is returned by the API as a model-dispersion diagnostic; it is not a calibrated confidence interval.
