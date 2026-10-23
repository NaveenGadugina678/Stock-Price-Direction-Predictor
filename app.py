# ============================================================
# Stock Price Direction Predictor — Flask Web App
# ============================================================
# Pages:
#   /           → Dashboard: today's signal + account summary
#   /trades     → Full trade history table
#   /portfolio  → P&L chart + performance stats
#   /run        → Trigger live_predictor manually from browser
# ============================================================

from flask import Flask, render_template, jsonify, redirect, url_for
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
import os
import subprocess
from datetime import datetime, timedelta

app = Flask(__name__)

# ── Config ───────────────────────────────────────────────────
API_KEY    = "xxxxxxxxxxxxxxxxxxxxxxxxxx"       # replace with your Key ID
SECRET_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # replace with your Secret Key ID
BASE_URL   = "https://paper-api.alpaca.markets"
SYMBOL     = "AMZN"
LOG_FILE   = "trades_log.csv"

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")


# ── Helper: get account info ──────────────────────────────────
def get_account():
    acc = api.get_account()
    return {
        "status"         : acc.status,
        "cash"           : f"${float(acc.cash):,.2f}",
        "portfolio_value": f"${float(acc.portfolio_value):,.2f}",
        "buying_power"   : f"${float(acc.buying_power):,.2f}",
        "pnl"            : float(acc.portfolio_value) - 100_000,
        "pnl_str"        : f"${float(acc.portfolio_value) - 100_000:+,.2f}",
        "pnl_positive"   : float(acc.portfolio_value) >= 100_000
    }


# ── Helper: get open position ─────────────────────────────────
def get_position():
    try:
        p = api.get_position(SYMBOL)
        return {
            "held"        : True,
            "qty"         : p.qty,
            "avg_entry"   : f"${float(p.avg_entry_price):.2f}",
            "current"     : f"${float(p.current_price):.2f}",
            "unrealized"  : f"${float(p.unrealized_pl):+.2f}",
            "is_positive" : float(p.unrealized_pl) >= 0
        }
    except:
        return {"held": False}


# ── Helper: get today's signal from log ───────────────────────
def get_latest_signal():
    if not os.path.exists(LOG_FILE):
        return None
    df = pd.read_csv(LOG_FILE, parse_dates=["date"])
    if df.empty:
        return None
    row = df.iloc[-1]
    return {
        "date"        : str(row["date"])[:10],
        "close"       : f"${float(row['close']):.2f}",
        "signal_label": row["signal_label"],
        "signal"      : int(row["signal"]),
        "confidence"  : f"{float(row['confidence']):.1f}%",
        "action"      : row["action"]
    }


# ── Helper: load trade history ────────────────────────────────
def get_trade_history():
    if not os.path.exists(LOG_FILE):
        return []
    df = pd.read_csv(LOG_FILE, parse_dates=["date"])
    df["date"]  = df["date"].astype(str).str[:10]
    df["close"] = df["close"].apply(lambda x: f"${float(x):.2f}")
    return df[["date","close","signal_label","confidence","action"]].to_dict("records")


# ── Helper: portfolio chart data ──────────────────────────────
def get_chart_data():
    if not os.path.exists(LOG_FILE):
        return [], [], []
    df = pd.read_csv(LOG_FILE, parse_dates=["date"])
    df["date"] = df["date"].astype(str).str[:10]
    dates  = df["date"].tolist()
    closes = df["close"].tolist()
    # Signal colors: 1=green, -1=red, 0=gray
    colors = ["#16a34a" if s == 1 else "#dc2626" if s == -1 else "#6b7280"
              for s in df["signal"].tolist()]
    return dates, closes, colors


# ── Routes ───────────────────────────────────────────────────

@app.route("/")
def dashboard():
    account = get_account()
    position = get_position()
    signal  = get_latest_signal()
    dates, closes, colors = get_chart_data()
    return render_template("dashboard.html",
        account=account,
        position=position,
        signal=signal,
        dates=dates,
        closes=closes,
        colors=colors,
        symbol=SYMBOL
    )


@app.route("/trades")
def trades():
    history = get_trade_history()
    return render_template("trades.html", trades=history, symbol=SYMBOL)


@app.route("/portfolio")
def portfolio():
    account  = get_account()
    position = get_position()
    history  = get_trade_history()
    total    = len(history)
    buys     = sum(1 for t in history if t["action"] == "BUY")
    sells    = sum(1 for t in history if t["action"] == "SELL")
    return render_template("portfolio.html",
        account=account,
        position=position,
        total=total,
        buys=buys,
        sells=sells,
        symbol=SYMBOL
    )


@app.route("/run")
def run_predictor():
    """Trigger live_predictor.py from the browser."""
    try:
        result = subprocess.run(
            ["python3", "live_predictor.py"],
            capture_output=True, text=True, timeout=60
        )
        output = result.stdout + result.stderr
    except Exception as e:
        output = str(e)
    return render_template("run.html", output=output)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
