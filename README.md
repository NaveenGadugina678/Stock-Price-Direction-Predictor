# 📈 Stock Price Direction Predictor

A Machine Learning project that predicts whether a stock's price will move UP 📈 or DOWN 📉 on the next trading day using historical stock market data and technical indicators. The project focuses on binary classification rather than exact price prediction, making it useful for trend analysis and trading decision support.

## 🚀 Project Overview

Predicting stock market behavior is a challenging problem due to market volatility and noise. This project leverages historical stock data along with engineered technical indicators to train a machine learning model capable of forecasting the next day's price direction.

### Objective
 
Given historical stock information:

Open Price
High Price
Low Price
Close Price
Volume
Technical Indicators

Predict:

Will the stock price go UP or DOWN tomorrow?

## 🔍 Workflow
1. Data Collection & Preparation
Historical stock data is collected.
Missing values are handled.
Data is cleaned and transformed for modeling.
2. Feature Engineering

Technical indicators and derived features are generated from historical price data.

Examples may include:

Moving Averages
Price Returns
Rolling Statistics
Momentum Features
Volatility Measures
3. Data Scaling

Features are normalized using a scaler to improve model performance.

4. Model Training

The classification model is trained to predict stock direction:

Tomorrow Direction:
1 → Price Goes Up
0 → Price Goes Down
5. Evaluation

Performance is measured using:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix

## 📊 Results

The trained model successfully learns patterns from historical market data and predicts future price direction with promising performance.

Sample Output
Actual	Predicted
UP	UP
DOWN	DOWN
UP	DOWN

See:

model_result.png
Closing_Price.jpeg
features_overview.jpeg

for visual analysis and performance insights.

## 🛠️ Technologies Used
Python
NumPy
Pandas
Scikit-Learn
Matplotlib
Joblib/Pickle

## ⚙️ Installation

Clone the repository:

git clone https://github.com/NaveenGadugina678/Stock-Price-Direction-Predictor.git
cd Stock-Price-Direction-Predictor

Install dependencies:

pip install -r requirements.txt

## ▶️ Running the Project
Preprocess Data
python feature_engineering.py
Generate Features
python train_model.py
Train Model
python data_preprocessing.py


📚 Learning Outcomes

This project demonstrates:

Financial data preprocessing
Feature engineering
Machine Learning classification
Model evaluation techniques
Saving and loading trained models
End-to-end predictive analytics workflow

## ⚠️ Disclaimer

This project is for educational and research purposes only. Stock market predictions are inherently uncertain and should not be considered financial advice.

## 👨‍💻 Author

Naveen Gadugina

GitHub:
NaveenGadugina678
