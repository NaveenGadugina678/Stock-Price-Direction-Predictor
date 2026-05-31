import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Ticker = "TCS.NS"
Start = "2020-01-01"
End = "2024-12-31"
Output = "Stock_data_processed.csv"

print(f"Downloading data for {Ticker}")
df = yf.download(Ticker, start = Start, end = End, auto_adjust = True)

if df.empty:
    raise ValueError(f"No data returned for {Ticker}. check ticker symbol.")

print(f"\nShape : {df.shape}")
print(f"\nColumns : {df.columns}")


