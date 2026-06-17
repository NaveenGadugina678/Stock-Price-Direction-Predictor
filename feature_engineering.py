import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

#----Loading the processed data----

df = pd.read_csv("stock_data_processed.csv", index_col= 0, parse_dates = True)
df.index = pd.to_datetime(df.index)
df = df.dropna()
print(f"\nLoad data shape : {df.shape}")
print(f"\nColumns : {list(df.columns)}")
print(f"\nFirst few rows :\n{df.head()}")

#----Selecting Features and Target----

FEATURE_COLS = ["SMA_5", "SMA_20", "Momentum_5", "Volatility_10", "RSI_14", "Volume_Change"]
TARGET_COL = "Target"

X = df[FEATURE_COLS].values  #2D Array, Rows = Days, Cols = features
y = df[TARGET_COL].values #1D Array

print(f"Feature matrix X shape : {X.shape} -> ({X.shape[0]}days, {X.shape[1]}features)")
print(f"Target y shape : {y.shape}")
print(f"Class balance -> Up days : {y.mean()*100:.1f}% | Down days : {(1-y.mean())*100:.1f}%")
      
#----Train and Test split----

split_idx = int(len(X)*0.8)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train size : {X_train.shape[0]} days")
print(f"Test size : {X_test.shape[0]} days")
print(f"Train date range : {df.index[0].date()} -> {df.index[split_idx-1].date()}")
print(f"Test date range :  {df.index[split_idx].date()} -> {df.index[-1].date()}")

#----Feature Scaling----

#StandardScaler converts each feature to mean = 0 , std = 1.
#Without scaling RSI would dominate just because it's large.
#KEY_RULE : fit scaler only on Train, then transform both train and test.
#Fitting on test would leak test data.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train) #learning mean and std.
X_test_scaled = scaler.transform(X_test) #applying same mean and std to test.

print(f"\nAfter scaling - Train feature means : {X_train_scaled.mean(axis = 0).round(2)}")
print(f"After scaling - Train feature stds : {X_train_scaled.std(axis = 0).round(2)}")

#----check----

feature_df = pd.DataFrame(X_train_scaled[:5], columns = FEATURE_COLS)
print("\nFirst five rows of scaled training Data : ")
print(feature_df.round(3).to_string())

#----Saving----

np.save("X_train.npy", X_train_scaled)
np.save("X_test.npy", X_test_scaled)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("All files are saved.")
