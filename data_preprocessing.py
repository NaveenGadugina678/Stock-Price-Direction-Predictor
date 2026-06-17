from datetime import timedelta
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#----Assinging----

Ticker = "MRF.NS"
Start = "2020-01-01"
End = "2024-12-31"
Output = "Stock_data_processed.csv"

#----Downloading----

print(f"Downloading data for {Ticker}")
df = yf.download(Ticker, start = Start, end = End, auto_adjust = True)

if df.empty:
    raise ValueError(f"No data returned for {Ticker}. check ticker symbol.")

print(f"\nShape : {df.shape}")
print(f"\nColumns : {list(df.columns)}")
print(f"\nFirst 5 rows : \n{df.head()}")
print(f"\nMissing values : \n{df.isnull().sum()}")
print(f"\nBasic stats : \n{df['Close'].describe()}")

#----Plotting Closing Price----

plt.figure(figsize=(12, 4))
plt.plot(df.index, df['Close'], linewidth = 1.2, color = "#2563eb")
plt.title(f"{Ticker} - Closing Price", fontsize = 14)
plt.xlabel("Date")
plt.ylabel("Price(INR)")
plt.tight_layout()
plt.savefig("Closing_Price.jpeg", dpi = 120)
plt.show()
print("Chart saved.")

#----Feature Engineering----

close = df["Close"].squeeze()
df["SMA_5"] = close.rolling(window = 5).mean()
df["SMA_20"] = close.rolling(window = 20).mean()

df["Momentum_5"] = close - close.shift(5)

df["Volatility_10"] = close.rolling(window = 10).std()

def compute_rsi(series, period = 14):
    delta = series.diff()
    gain = delta.clip(lower = 0)
    loss = -delta.clip(upper = 0)
    avg_gain = gain.rolling(window = period, min_periods = period).mean()
    avg_loss = loss.rolling(window = period, min_periods = period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

df["RSI_14"] = compute_rsi(close)
df["Volume_Change"] = df["Volume"].pct_change()

#----Creating the target label----

# 1 - next day's close is HIGHER than today's close.
# 0 - next day's close is LOWER than today's close.

df["Target"] = (close.shift(-1)>close).astype(int)
df.dropna(inplace = True)

#----Plotting features----

print(f"\nClass distribution (0 = DOWN, 1 = UP): ")
print(df["Target"].value_counts())
print(f"Updays : {df["Target"].mean()*100:.1f}%")

#----Plot features----

fig, axes = plt.subplots(3, 1, figsize = (12, 8), sharex = True )

axes[0].plot(df.index, df["Close"], label = "Close", linewidth = 1)
axes[0].plot(df.index, df["SMA_5"], label = "SMA 5:, linewidth = 1", linestyle = "--")
axes[0].plot(df.index, df["SMA_20"], label = "SMA 20:, linewidth = 1", linestyle = ":")
axes[0].set_ylabel("Price")
axes[0].legend(fontsize = 8)
axes[0].set_title("Price and Moving averages")

axes[1].plot(df.index, df["RSI_14"], color = "darkorange", linewidth = 1)
axes[1].axhline(70, color = "red", linestyle = "--", linewidth = 0.8, label = "Overbought(70)")
axes[1].axhline(30, color = "green", linestyle = "--", linewidth = 0.8, label = "Oversold(30)")
axes[1].set_ylabel("RSI")
axes[1].legend(fontsize = 8)
axes[1].set_title("RSI(14-Day)")

axes[2].bar(df.index, df["Volume"].squeeze(), color = "#93c5fd", width = timedelta(days = 1)) 
axes[2].set_ylabel("Volume")
axes[2].set_title("Trading Volume")

plt.tight_layout()
plt.savefig("features_overview.jpeg", dpi = 120)
plt.show()
print("Features chart saved")

#----Saving Processed Data----

features_cols = ["Close", "SMA_5", "SMA_20", "Momentum_5", "Volatility_10", "RSI_14", "Volume_Change", "Target"]
df[features_cols].to_csv(Output)
print(f"Processed data saved to '{Output}'")
print(f"Final shape : {df[features_cols].shape}")
