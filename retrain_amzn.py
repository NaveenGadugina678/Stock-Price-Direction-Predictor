# ============================================================
# Retrain full pipeline on AMZN data
# ============================================================
# Runs the complete Day1→Day5 pipeline on Amazon stock data
# and saves best_model_v2.pkl, scaler_v2.pkl, feature_cols.pkl
# These are what live_predictor.py loads every day.
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, balanced_accuracy_score
warnings.filterwarnings("ignore")

TICKER    = "AMZN"
START     = "2018-01-01"
END       = "2024-12-31"
THRESHOLD = 0.5    # predict only moves > 0.5%


# ── Step 1: Download AMZN data ───────────────────────────────
print(f"Downloading {TICKER} data...")
df = yf.download(TICKER, start=START, end=END, auto_adjust=True)
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
close  = df["Close"].squeeze()
volume = df["Volume"].squeeze()
print(f"Downloaded {len(df)} days of data.")


# ── Step 2: Compute RSI ──────────────────────────────────────
def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs       = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


# ── Step 3: Feature engineering ──────────────────────────────
print("Engineering features...")

feat = pd.DataFrame(index=df.index)
feat["SMA_5"]          = close.rolling(5).mean()
feat["SMA_20"]         = close.rolling(20).mean()
feat["Momentum_5"]     = close - close.shift(5)
feat["Volatility_10"]  = close.rolling(10).std()
feat["RSI_14"]         = compute_rsi(close)
feat["Volume_Change"]  = volume.pct_change()
feat["Price_vs_SMA20"] = (close - feat["SMA_20"]) / feat["SMA_20"] * 100
feat["SMA_Cross"]      = feat["SMA_5"] - feat["SMA_20"]
feat["Daily_Return"]   = close.pct_change() * 100

# Lag features
for col in ["RSI_14", "Momentum_5", "Volume_Change"]:
    for lag in [1, 2, 3]:
        feat[f"{col}_lag{lag}"] = feat[col].shift(lag)


# ── Step 4: Smart target (only significant moves) ────────────
tomorrow_return   = (close.shift(-1) - close) / close * 100
feat["Target"]    = 0
feat.loc[tomorrow_return >  THRESHOLD, "Target"] =  1
feat.loc[tomorrow_return < -THRESHOLD, "Target"] = -1

# Keep only signal days
feat = feat[feat["Target"] != 0].copy()
feat.dropna(inplace=True)

FEATURE_COLS = [c for c in feat.columns if c != "Target"]
print(f"Signal days : {len(feat)}  |  Features: {len(FEATURE_COLS)}")
print(f"Up: {(feat['Target']==1).sum()}  |  Down: {(feat['Target']==-1).sum()}")


# ── Step 5: Train/test split ─────────────────────────────────
X = feat[FEATURE_COLS].values
y = feat["Target"].values

split_idx   = int(len(X) * 0.8)
X_train_raw = X[:split_idx]
X_test_raw  = X[split_idx:]
y_train     = y[:split_idx]
y_test      = y[split_idx:]

scaler      = StandardScaler()
X_train     = scaler.fit_transform(X_train_raw)
X_test      = scaler.transform(X_test_raw)

print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")


# ── Step 6: Train model ───────────────────────────────────────
print("Training Random Forest on AMZN data...")

model = RandomForestClassifier(
    n_estimators    = 200,
    max_depth       = 5,
    min_samples_leaf= 10,
    class_weight    = "balanced",
    random_state    = 42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n── AMZN Model Results ────────────────────────────────")
print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred)*100:.1f}%")


# ── Step 7: Feature importance ───────────────────────────────
print("\n── Top 5 Important Features ──────────────────────────")
importances = model.feature_importances_
top5_idx    = np.argsort(importances)[::-1][:5]
for i in top5_idx:
    print(f"  {FEATURE_COLS[i]:<22} {importances[i]:.4f}")


# ── Step 8: Save everything ───────────────────────────────────
with open("best_model_v2.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler_v2.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_cols.pkl", "wb") as f:
    pickle.dump(FEATURE_COLS, f)

print("\n✓ Saved: best_model_v2.pkl, scaler_v2.pkl, feature_cols.pkl")
print("You can now run live_predictor.py safely on AMZN!")
