# ====================
# Alpaca Paper Trading 
# ====================

import alpaca_trade_api as tradeapi
import pandas as pd
import matplotlib.pyplot as plt
import os

API_KEY    = "xxxxxxxxxxxxxxxxxxxxxxxxxx"
SECRET_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
BASE_URL   = "https://paper-api.alpaca.markets"
LOG_FILE   = "trades_log.csv"

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")


# ── Live account snapshot ─────────────────────────────────────
account = api.get_account()
print("── Live Account Snapshot ─────────────────────────────")
print(f"Portfolio value : ${float(account.portfolio_value):,.2f}")
print(f"Cash            : ${float(account.cash):,.2f}")
print(f"Buying power    : ${float(account.buying_power):,.2f}")

# P&L vs starting $100k
pnl = float(account.portfolio_value) - 100_000
print(f"Total P&L       : ${pnl:+,.2f}")


# ── Open positions ────────────────────────────────────────────
try:
    positions = api.list_positions()
    if positions:
        print("\n── Open Positions ────────────────────────────────────")
        for p in positions:
            unrealized = float(p.unrealized_pl)
            print(f"{p.symbol}: {p.qty} shares | "
                  f"Avg entry: ${float(p.avg_entry_price):.2f} | "
                  f"Current: ${float(p.current_price):.2f} | "
                  f"Unrealized P&L: ${unrealized:+.2f}")
    else:
        print("\nNo open positions.")
except Exception as e:
    print(f"\nCould not fetch positions: {e}")


# ── Trade log analysis ────────────────────────────────────────
if not os.path.exists(LOG_FILE):
    print(f"\nNo trade log found yet. Run live_predictor.py first.")
else:
    df = pd.read_csv(LOG_FILE, parse_dates=["date"])
    print(f"\n── Trade Log Summary ({LOG_FILE}) ────────────────────")
    print(f"Total signals logged : {len(df)}")

    buys  = df[df["action"] == "BUY"]
    sells = df[df["action"] == "SELL"]
    holds = df[df["action"].isin(["HOLD", "SKIP"])]

    print(f"BUY  signals : {len(buys)}")
    print(f"SELL signals : {len(sells)}")
    print(f"HOLD/SKIP    : {len(holds)}")

    up_signals   = (df["signal"] == 1).sum()
    down_signals = (df["signal"] == -1).sum()
    print(f"\nUp predictions   : {up_signals}")
    print(f"Down predictions : {down_signals}")

    avg_confidence = df["confidence"].mean()
    print(f"Avg confidence   : {avg_confidence:.1f}%")

    print(f"\n── Full Trade History ────────────────────────────────")
    display_cols = ["date", "close", "signal_label", "confidence", "action"]
    print(df[display_cols].to_string(index=False))


    # ── Performance chart ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # AMZN close price over logged days
    axes[0].plot(df["date"], df["close"], color="#2563eb", linewidth=1.5, marker="o", markersize=4)
    buy_days  = df[df["action"] == "BUY"]
    sell_days = df[df["action"] == "SELL"]
    axes[0].scatter(buy_days["date"],  buy_days["close"],  color="#16a34a", s=80, zorder=5, label="BUY",  marker="^")
    axes[0].scatter(sell_days["date"], sell_days["close"], color="#dc2626", s=80, zorder=5, label="SELL", marker="v")
    axes[0].set_title(f"AMZN Price + Trade Signals")
    axes[0].set_ylabel("Price ($)")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)

    # Signal distribution
    signal_counts = df["action"].value_counts()
    colors = {"BUY": "#16a34a", "SELL": "#dc2626", "HOLD": "#d97706", "SKIP": "#6b7280"}
    bar_colors = [colors.get(a, "#888") for a in signal_counts.index]
    axes[1].bar(signal_counts.index, signal_counts.values, color=bar_colors, alpha=0.85)
    axes[1].set_title("Action Distribution")
    axes[1].set_ylabel("Count")

    plt.suptitle("Paper Trading Performance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("performance.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("\nChart saved as performance.png")

print("\nDone!")
