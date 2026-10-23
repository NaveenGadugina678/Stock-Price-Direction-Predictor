import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, balanced_accuracy_score
import pickle, warnings
warnings.filterwarnings("ignore")


# ── Reload and rebuild with smarter target ───────────
df = pd.read_csv("stock_data_processed.csv", index_col=0, parse_dates=True)
close = df["Close"]

# OLD target: up or down by ANY amount (too noisy)
# NEW target: up by more than THRESHOLD % (meaningful move only)
THRESHOLD = 0.49   

tomorrow_return = (close.shift(-1) - close) / close * 100

# 3-class target: 1=significant up, -1=significant down, 0=flat (ignored)
df["Return_Tomorrow"] = tomorrow_return
df["Target_Smart"] = 0
df.loc[tomorrow_return >  THRESHOLD, "Target_Smart"] =  1
df.loc[tomorrow_return < -THRESHOLD, "Target_Smart"] = -1
# Only keep rows where there IS a signal (drop flat days)
df_signal = df[df["Target_Smart"] != 0].copy()
print(f"Total days: {len(df)}  |  Signal days (non-flat): {len(df_signal)}")
print(f"Up signals:   {(df_signal['Target_Smart']==1).sum()}")
print(f"Down signals: {(df_signal['Target_Smart']==-1).sum()}")


# ── Add lag features (model memory) ──────────────────
# Lag features give the model information about what happened
# yesterday, 2 days ago, 3 days ago — like short-term memory.

base_features = ["RSI_14", "Momentum_5", "Volatility_10",
                 "SMA_5", "SMA_20", "Volume_Change"]

for feat in ["RSI_14", "Momentum_5", "Volume_Change"]:
    for lag in [1, 2, 3]:
        df_signal[f"{feat}_lag{lag}"] = df_signal[feat].shift(lag)

df_signal.dropna(inplace=True)

FEATURE_COLS = [c for c in df_signal.columns
                if c not in ["Target_Smart", "Return_Tomorrow", "Target", "Close"]]
print(f"\nTotal features with lags: {len(FEATURE_COLS)}")
print(f"Dataset after filtering:  {df_signal.shape}")


# ── Train/test split ─────────────────────────────────
X = df_signal[FEATURE_COLS].values
y = df_signal["Target_Smart"].values

split_idx   = int(len(X) * 0.8)
X_train_raw = X[:split_idx]
X_test_raw  = X[split_idx:]
y_train     = y[:split_idx]
y_test      = y[split_idx:]

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")


# ── Train model ──────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("\n── Results ──")
print(classification_report(y_test, y_pred, target_names=["Down","Up"]))
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred)*100:.1f}%")


# ── Backtest — simulate actual trading ───────────────
# Strategy: if model predicts Up → buy (go long)
#           if model predicts Down → sell (go short)
#           flat prediction → do nothing

test_df = df_signal.iloc[split_idx:].copy()
test_df["Predicted"]      = y_pred
test_df["Actual_Return"]  = test_df["Return_Tomorrow"]

# Trade return: if prediction matches direction, you profit
test_df["Trade_Return"] = test_df.apply(
    lambda row: row["Actual_Return"]  if row["Predicted"] ==  1 else
               -row["Actual_Return"]  if row["Predicted"] == -1 else 0,
    axis=1
)

# Cumulative returns
test_df["Cumulative_Strategy"]   = (1 + test_df["Trade_Return"] / 100).cumprod()
test_df["Cumulative_BuyAndHold"] = (1 + test_df["Actual_Return"] / 100).cumprod()

total_trades    = (test_df["Predicted"] != 0).sum()
winning_trades  = ((test_df["Trade_Return"] > 0)).sum()
win_rate        = winning_trades / total_trades * 100 if total_trades > 0 else 0
strategy_return = (test_df["Cumulative_Strategy"].iloc[-1] - 1) * 100
buyhold_return  = (test_df["Cumulative_BuyAndHold"].iloc[-1] - 1) * 100

print(f"\n── Backtest Results ──")
print(f"Total trades    : {total_trades}")
print(f"Win rate        : {win_rate:.1f}%")
print(f"Strategy return : {strategy_return:.1f}%")
print(f"Buy & Hold      : {buyhold_return:.1f}%")
print(f"Alpha (edge)    : {strategy_return - buyhold_return:.1f}%")


# ── Plot ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Backtest curve
axes[0].plot(test_df.index, test_df["Cumulative_Strategy"],
             label="Our Strategy", color="#2563eb", linewidth=1.5)
axes[0].plot(test_df.index, test_df["Cumulative_BuyAndHold"],
             label="Buy & Hold", color="#dc2626", linewidth=1.5, linestyle="--")
axes[0].axhline(1, color="gray", linestyle=":", linewidth=0.8)
axes[0].set_title("Backtest: Strategy vs Buy & Hold")
axes[0].set_ylabel("Portfolio value (starting = 1.0)")
axes[0].legend()

# Feature importance
importances = rf.feature_importances_
top_n       = 10
top_idx     = np.argsort(importances)[-top_n:]
axes[1].barh([FEATURE_COLS[i] for i in top_idx],
             importances[top_idx], color="#16a34a", alpha=0.8)
axes[1].set_title("Top 10 Feature Importances")
axes[1].set_xlabel("Importance")

plt.suptitle("Smarter Target + Backtest", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("smart_approach_results.png", dpi=130, bbox_inches="tight")
plt.show()
print("\nSaved: smart_approach_results.png")

# ── Save model ──────────────────────────────────────
with open("best_model_v2.pkl", "wb") as f:
    pickle.dump(rf, f)
with open("scaler_v2.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("feature_cols.pkl", "wb") as f:
    pickle.dump(FEATURE_COLS, f)

print("Saved: best_model_v2.pkl, scaler_v2.pkl, feature_cols.pkl")
