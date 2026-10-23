# ====================
# Alpaca Paper Trading
# ====================

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta

# ── Config ───────────────────────────────────────────────────
API_KEY    = "xxxxxxxxxxxxxxxxxxxxxxxxxx"
SECRET_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
BASE_URL   = "https://paper-api.alpaca.markets"
SYMBOL     = "AMZN"
TRADE_QTY  = 1          # number of shares per trade (start small)
LOG_FILE   = "trades_log.csv"

# ── Connect to Alpaca ─────────────────────────────────────────
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")


# ── Helper: compute RSI ───────────────────────────────────────
def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs       = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


# ── Step 1: Fetch recent AMZN bars from Alpaca ───────────────
print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching AMZN data...")

end   = datetime.now()
start = end - timedelta(days=90)   # last 90 days for feature computation

bars = api.get_bars(
    SYMBOL,
    tradeapi.rest.TimeFrame.Day,
    start.strftime("%Y-%m-%d"),
    end.strftime("%Y-%m-%d"),
    adjustment="raw",
    feed="iex"
).df

if bars.empty or len(bars) < 30:
    print("Not enough data to compute features. Try again tomorrow.")
    exit()

bars.index = pd.to_datetime(bars.index)
close  = bars["close"]
volume = bars["volume"]

print(f"Got {len(bars)} days of data. Latest close: ${close.iloc[-1]:.2f}")


# ── Step 2: Compute features (same as training) ───────────────
print("Computing features...")

df = pd.DataFrame(index=bars.index)
df["Close"]          = close
df["SMA_5"]          = close.rolling(5).mean()
df["SMA_20"]         = close.rolling(20).mean()
df["Momentum_5"]     = close - close.shift(5)
df["Volatility_10"]  = close.rolling(10).std()
df["RSI_14"]         = compute_rsi(close)
df["Volume_Change"]  = volume.pct_change()
df["Price_vs_SMA20"] = (close - df["SMA_20"]) / df["SMA_20"] * 100
df["SMA_Cross"]      = df["SMA_5"] - df["SMA_20"]
df["Daily_Return"]   = close.pct_change() * 100

# Lag features
for feat in ["RSI_14", "Momentum_5", "Volume_Change"]:
    for lag in [1, 2, 3]:
        df[f"{feat}_lag{lag}"] = df[feat].shift(lag)

df.dropna(inplace=True)

# Load feature column order from training (must match exactly)
with open("feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# Get today's feature row (last row)
today_features = df[feature_cols].iloc[-1].values.reshape(1, -1)
today_date     = df.index[-1].strftime("%Y-%m-%d")
today_close    = close.iloc[-1]

print(f"Features computed for {today_date}")


# ── Step 3: Load model + scaler and predict ───────────────────
print("Loading model...")

with open("best_model_v2.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler_v2.pkl", "rb") as f:
    scaler = pickle.load(f)

today_scaled = scaler.transform(today_features)
signal       = model.predict(today_scaled)[0]
proba        = model.predict_proba(today_scaled)[0]

signal_label = "UP ↑" if signal == 1 else "DOWN ↓"
confidence   = max(proba) * 100

print(f"\n── Today's Signal ────────────────────────────────────")
print(f"Date       : {today_date}")
print(f"AMZN close : ${today_close:.2f}")
print(f"Signal     : {signal_label}")
print(f"Confidence : {confidence:.1f}%")


# ── Step 4: Check existing positions ─────────────────────────
try:
    position = api.get_position(SYMBOL)
    current_qty = int(position.qty)
    print(f"\nCurrent position: {current_qty} shares of {SYMBOL}")
except:
    current_qty = 0
    print(f"\nNo current position in {SYMBOL}")


# ── Step 5: Place order based on signal ──────────────────────
order_placed = None
order_side   = None

print(f"\n── Order Decision ────────────────────────────────────")

if signal == 1 and current_qty == 0:
    # Signal is Up and we have no position → BUY
    print(f"Placing BUY order: {TRADE_QTY} share(s) of {SYMBOL}")
    order = api.submit_order(
        symbol     = SYMBOL,
        qty        = TRADE_QTY,
        side       = "buy",
        type       = "market",
        time_in_force = "day"
    )
    order_placed = order.id
    order_side   = "BUY"
    print(f"✓ BUY order placed! Order ID: {order.id}")

elif signal == -1 and current_qty > 0:
    # Signal is Down and we hold shares → SELL
    print(f"Placing SELL order: {current_qty} share(s) of {SYMBOL}")
    order = api.submit_order(
        symbol     = SYMBOL,
        qty        = current_qty,
        side       = "sell",
        type       = "market",
        time_in_force = "day"
    )
    order_placed = order.id
    order_side   = "SELL"
    print(f"✓ SELL order placed! Order ID: {order.id}")

elif signal == 1 and current_qty > 0:
    print("Signal is UP but we already hold shares → HOLD, no order.")
    order_side = "HOLD"

elif signal == -1 and current_qty == 0:
    print("Signal is DOWN and we have no position → SKIP, no order.")
    order_side = "SKIP"


# ── Step 6: Log the trade ─────────────────────────────────────
log_entry = {
    "date"       : today_date,
    "symbol"     : SYMBOL,
    "close"      : today_close,
    "signal"     : signal,
    "signal_label": signal_label,
    "confidence" : round(confidence, 2),
    "action"     : order_side,
    "order_id"   : order_placed if order_placed else "N/A",
    "qty"        : TRADE_QTY,
    "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

log_df = pd.DataFrame([log_entry])

if os.path.exists(LOG_FILE):
    log_df.to_csv(LOG_FILE, mode="a", header=False, index=False)
else:
    log_df.to_csv(LOG_FILE, index=False)

print(f"\n✓ Logged to {LOG_FILE}")
print("\nDone! Run this again tomorrow for the next signal.")
