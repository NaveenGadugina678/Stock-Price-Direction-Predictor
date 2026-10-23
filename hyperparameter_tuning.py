import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import(
        balanced_accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import pickle, warnings
warnings.filterwarnings("ignore")

FEATURE_COLS_OLD = ["SMA_5", "SMA_20", "Momentum_5", "Volatility_10", "RSI_14", "Volume_Change"]

#----Reloading raw data and engineering better features----

df = pd.read_csv("stock_data_processed.csv", index_col = 0, parse_dates = True)
close = df["Close"]
volume = pd.read_csv("stock_data_processed.csv", index_col = 0, parse_dates = True)
df_feat = df[FEATURE_COLS_OLD].copy() #keeping old features.

#----New Features----
#1: Price distance from SMA_20 (%)

df_feat["Price_vs_SMA20"] = ((close - df["SMA_20"]) / df["SMA_20"]) * 100

#2: SMA crossover signal
#if SMA_5 crosses SMA_20 then -> bullish (Golden Cross)
#if SMA_20 crosses SMA_5 then -> bearish  (Death Cross)
df["SMA_Cross"] = df["SMA_5"] - df["SMA_20"]

#3: Price change % (today vs yesterday)
df_feat["Daily_Return"] = close.pct_change() * 100

#4: Volume Z-Score
vol_mean = df["Volume_Change"].rolling(20).mean() if "Volume Change" in df.columns else close.rolling(20).mean()
vol_std = df["Volume_Change"].rolling(20).std() if "Volume Change" in df.columns else close.rolling(20).std() 
df_feat["Vol_ZScore"] = (df["Volume_Change"] - vol_mean) / (vol_std + 1e-10)

df_feat["Target"] = df["Target"]
df_feat.dropna(inplace = True)

FEATURE_COLS_NEW = [c for c in df_feat.columns if c != "Target"]
print(f"Features ({len(FEATURE_COLS_NEW)}) : {FEATURE_COLS_NEW}")
print(f"Dataset shape : {df_feat.shape}")

#----Train and Test Split----
X = df_feat[FEATURE_COLS_NEW].values
y = df_feat["Target"].values

split_idx = int(len(X) * 0.8)
X_train_raw = X[:split_idx]
X_test_raw = X[split_idx + 1 :]
y_train = y[:split_idx]
y_test = y[split_idx + 1:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

print(f"Train Shape : {X_train.shape}, Test Shape : {X_test.shape}")
print(f"Class Balance -> Up : {y_train.mean()*100:.1f}% Down : {1 - y_train.mean()*100:.1f}%")

#----Logistic Regression----
print("\n----Logistic Regression with balanced class----")
lr_balanced = LogisticRegression(
        max_iter = 1000,
        random_state = 42,
        class_weight = "balanced"
        )
lr_balanced.fit(X_train, y_train)
y_pred_lr = lr_balanced.predict(X_test)

print(classification_report(y_test, y_pred_lr, target_names = ["Down", "Up"]))
print(f"Balanced Accuracy : {balanced_accuracy_score(y_test, y_pred_lr)*100:.1f}%")

#----Random Forest Classifier----
print("----Random Forest Tunning----")
tscv = TimeSeriesSplit(n_splits = 5)

param_grid = {
        "n_estimators" : [100, 200],
        "max_depth" : [4, 5, 6],
        "min_samples_leaf" : [10, 20],
        "class_weight" : ["balanced"]
        }

rf_grid = GridSearchCV(
        RandomForestClassifier(random_state = 42),
        param_grid,
        cv = tscv,
        scoring = "balanced_accuracy",
        n_jobs = -1,
        verbose = 0
        )

rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)

print(f"Best params : {rf_grid.best_params_}")
print(classification_report(y_test, y_pred_rf, target_names = ["Down", "Up"]))
print(f"Balanced Accureacy : {balanced_accuracy_score(y_test, y_pred_rf)*100:.1f}%")

#----Plot Results----
fig, axes = plt.subplots(1, 3, figsize = (15, 5))

#----Confusion Matrix : Linear Regression----
ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred_lr),
        display_labels = ["Down", "Up"]).plot(ax = axes[0], colorbar = False, cmap = "Blues")
axes[0].set_title(f"Logistic Regression (Balanced)\nBalanced Acc : {balanced_accuracy_score(y_test, y_pred_lr)*100:.1f}%")

#----Confusion Matrix : Random Forest Classifier----
ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred_rf),
        display_labels = ["Down", "Up"]).plot(ax = axes[1], colorbar = False, cmap = "Greens")
axes[1].set_title(f"Random Forest (Tuned)\nBalanced Acc : {balanced_accuracy_score(y_test, y_pred_rf)*100:.1f}%")

#----Feature Importance----
importances = best_rf.feature_importances_
sorted_idx = np.argsort(importances)
axes[2].barh(
        [FEATURE_COLS_NEW[i] for i in sorted_idx],
        importances[sorted_idx],
        color = "#16a34a", alpha = 0.8
        )
axes[2].set_title("Feature Importance\n(Tuned Random Forest)")
axes[2].set_xlabel("Importance")

plt.suptitle("Fixed Model Results", fontsize = 13, fontweight = "bold")
plt.tight_layout()
plt.savefig("Tuned_results.png", dpi = 130, bbox_inches = "tight")
plt.show()

#----Saving Improved Model----

with open("best_model_v2.pkl", "wb") as f:
    pickle.dump(best_rf, f)
with open("scaler_v2.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("feature_cols.pkl", "wb") as f:
    pickle.dump(FEATURE_COLS_NEW, f)

