import io
import requests
import yfinance as yf
import numpy as np
import pandas as pd

# 1. Fetch S&P 500 tickers dynamically from Wikipedia
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
sp500_table = pd.read_html(io.StringIO(response.text))[0]

# Yahoo Finance uses '-' instead of '.' (e.g., BRK-B instead of BRK.B)
tickers = sp500_table["Symbol"].str.replace(".", "-", regex=False).tolist()

# Maps for company name and sector
ticker_to_name = dict(zip(tickers, sp500_table["Security"]))
ticker_to_sector = dict(zip(tickers, sp500_table["GICS Sector"]))

print(f"Loaded {len(tickers)} tickers. Downloading price data...")

# 2. Download 5 years of daily adjusted prices
prices = yf.download(
    tickers,
    start="2021-09-30",
    end="2026-09-30",
    auto_adjust=True,
    progress=True
)["Close"]

# Drop any stocks that don't have full 5-year history / have missing data
prices = prices.dropna(axis=1)

# 3. Standard (simple) daily returns
returns = prices.pct_change()

# 4. Annualized volatility
volatility = returns.std() * np.sqrt(252)

# 5. Rank from lowest to highest
volatility = volatility.sort_values()

# 30 lowest-volatility stocks
lowest_30 = volatility.head(30)

summary_df = pd.DataFrame({
    "Rank": range(1, 31),
    "Ticker": lowest_30.index,
    "Company": [ticker_to_name.get(t, "N/A") for t in lowest_30.index],
    "Sector": [ticker_to_sector.get(t, "N/A") for t in lowest_30.index],
    "Annualized Vol": lowest_30.values.round(4),
    "Vol (%)": (lowest_30.values * 100).round(2).astype(str) + "%"
})

print("\nTop 30 Lowest Volatility S&P 500 Stocks (5-Year):")
print(summary_df.to_string(index=False))
