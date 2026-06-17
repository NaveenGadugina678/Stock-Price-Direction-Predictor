import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay)
import pickle

FEATURE_COlS = ["SMA_5", "SMA_20", "Momentum_5", "Volatility_10", "RSI_14", "Volume_Change"]

#----Loading data----

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print(f"Train : {X_train.shape}, Test : {X_test.shape}")

#----Train Model-1 = Logistic Regression

lr = LogisticRegression(max_iter = 1000, random_state = 42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("\n----Logistic Regression----")
print(f"    Accuracy : {accuracy_score(y_test, y_pred_lr)*100:.1f}%")
print(f"    Precision : {precision_score(y_test, y_pred_lr)*100:.1f}%")
print(f"    Recall : {recall_score(y_test, y_pred_lr)*100:.1f}%")
print(f"    F1 score : {f1_score(y_test, y_pred_lr)*100:.1f}%")

#----Random Forest----

rf = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n----Random Forest Classifier----")
print(f"    Accuracy : {accuracy_score(y_test, y_pred_rf)*100:.1f}%")
print(f"    Precision : {precision_score(y_test, y_pred_rf)*100:.1f}%")
print(f"    Recall : {recall_score(y_test, y_pred_rf)*100:.1f}%")
print(f"    F1 score : {f1_score(y_test, y_pred_rf)*100:.1f}%")

#----Baseline----
baseline_acc = max(y_test.mean(), 1-y_test.mean())*100
print(f"\nBaseline : {baseline_acc:.1f}%")

#----Plotting everything----

fig = plt.figure(figsize = (14, 10))
gs = gridspec.GridSpec(2, 3, figure = fig, hspace = 0.45, wspace = 0.35)

#----Bar Chart : model comparision----
ax_bar = fig.add_subplot(gs[0, :2])
metrics = ["Accuracy", "Precision", "Recall", "F1"]
lr_scores = [accuracy_score(y_test, y_pred_lr), precision_score(y_test, y_pred_lr), recall_score(y_test, y_pred_lr), f1_score(y_test, y_pred_lr)]
rf_scores = [accuracy_score(y_test, y_pred_rf), precision_score(y_test, y_pred_rf), recall_score(y_test, y_pred_rf), f1_score(y_test, y_pred_rf)]
x = np.arange(len(metrics))
w = 0.35
bars1 = ax_bar.bar(x - w/2, lr_scores, w, label = "Logistic Regression", color = "#2563eb", alpha = 0.85)
bars2 = ax_bar.bar(x + w/2, rf_scores, w, label = "Random Forest", color = "#16a34a", alpha = 0.85)
ax_bar.axhline(baseline_acc/100, color = "red", linestyle = "--", linewidth=1, label = f"Baseline ({baseline_acc*100:.0f}%)")
ax_bar.set_xticks(x); ax_bar.set_xticklabels(metrics)
ax_bar.set_ylim(0, 1); ax_bar.set_ylabel("Scores")
ax_bar.set_title("Model Comparision")
ax_bar.legend(fontsize = 9)

for bar in bars1:
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height()*100:.0f}%", ha = "center", fontsize = 8)

for bar in bars2:
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height()*100:.0f}%", ha = "center", fontsize = 8)

#----Confusion Matrix----

ax_cm1 = fig.add_subplot(gs[1, 0])
ax_cm2 = fig.add_subplot(gs[1, 1])
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_lr), display_labels = ["Down", "Up"]).plot(ax = ax_cm1, colorbar = False, cmap = "Blues")
ax_cm1.set_title("Logistic Regression\nConfusion Matrix")
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf), display_labels = ["Down", "Up"]).plot(ax = ax_cm2, colorbar = False, cmap = "Greens")
ax_cm2.set_title("Random Forest\nConfusion Matrix")

#----How to read Confusion Matrix----

ax_guide = fig.add_subplot(gs[1, 2])
ax_guide.axis("off")
guide = (
        "Reading the matrix : \n\n"
        "True Positive(TP) : \nPredicted Up was Up\n\n"
        "True Negative(TN) : \nPredicted Down was Down\n\n"
        "False Positive(FP) : \nPredicted Up was Down\n\n"
        "False Negative(FN) : \nPredicted Down was Up\n\n"
        "Precision : TP / (TP + FP)\n"
        "Recall : TP / (TP + FN)\n"
        )

ax_guide.text(0, 0.95, guide, transform = ax_guide.transAxes, fontsize = 9, verticalalignment = "top", bbox = dict(boxstyle = "round", facecolor = "#f0f9ff", alpha = 0.8))
plt.suptitle("Model Training Results", fontsize = 13, fontweight = "bold")
plt.savefig("model_result.png", dpi = 130, bbox_inches = "tight")
plt.show()
print("\nChart saved as model_result.png")


#----Saving the best model----
best_model = rf if f1_score(y_test, y_pred_rf) >= f1_score(y_test, y_pred_lr) else lr
best_name = "Random forest" if best_model is rf else "Logistic Regression"

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"\nBest model saved as best_model.pkl")
