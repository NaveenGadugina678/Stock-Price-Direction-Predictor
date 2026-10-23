# ============================================================
# Alpaca Paper Trading 
# ============================================================

import alpaca_trade_api as tradeapi

# ── Your Alpaca paper trading credentials ────────────────────

API_KEY    = "xxxxxxxxxxxxxxxxxxxxxxxxxx"       # replace with your Key ID
SECRET_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"    # replace with your Secret Key
BASE_URL   = "https://paper-api.alpaca.markets"   # paper trading URL

# ── Connect ──────────────────────────────────────────────────
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")

# ── Check account ────────────────────────────────────────────
account = api.get_account()

print("── Alpaca Paper Account ──────────────────────────────")
print(f"Status          : {account.status}")
print(f"Cash balance    : ${float(account.cash):,.2f}")
print(f"Portfolio value : ${float(account.portfolio_value):,.2f}")
print(f"Buying power    : ${float(account.buying_power):,.2f}")
print(f"Currency        : {account.currency}")

# ── Check AMZN is tradeable ──────────────────────────────────
asset = api.get_asset("AMZN")
print(f"\n── AMZN Asset Info ───────────────────────────────────")
print(f"Name        : {asset.name}")
print(f"Tradeable   : {asset.tradable}")
print(f"Fractionable: {asset.fractionable}")
print(f"Exchange    : {asset.exchange}")

if account.status == "ACTIVE" and asset.tradable:
    print("\n✓ Everything looks good! Ready to trade.")
else:
    print("\n✗ Something is off — check your keys or account status.")
