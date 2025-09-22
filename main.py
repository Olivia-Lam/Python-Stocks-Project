import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
# ---------------------------
# Download Stock Data Function
# ---------------------------
def download_stock_data(ticker, period='3y'):
    """
    Download stock data for the given ticker and period
    """
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

# ---------------------------
# Analysis Functions
# ---------------------------
def _compute_smas(prices, windows=(12, 50, 200)):
    """Efficient O(n) sliding-window SMAs using one cumulative sum pass."""
    s = prices.astype("float64")
    csum = s.cumsum()
    out = {}
    for w in windows:
        sma = (csum - csum.shift(w)) / w
        # enforce strict window: first w-1 are NaN
        if w > 1:
            sma.iloc[:w-1] = np.nan
        out[w] = sma
    return out


def simple_moving_average(ticker, period='3y'):
    data = download_stock_data(ticker, period)
    prices = data['Close']

    # --- O(n) sliding-window SMAs (12/50/200) ---
    smas = _compute_smas(prices, windows=(12, 50, 200))
    ma_12 = smas[12]
    ma_50 = smas[50]
    ma_200 = smas[200]

    fig, ax = plt.subplots(figsize=(14, 6), facecolor='#e6e6e6')
    ax.set_facecolor('#e6e6e6')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.tick_params(axis='y', right=True, left=False, colors='#222')
    ax.tick_params(axis='x', colors='#222')

    ax.plot(prices.index, prices, linewidth=1.2, label='Close')
    ax.plot(ma_12.index, ma_12, linewidth=1.2, label='SMA 12')
    ax.plot(ma_50.index, ma_50, linewidth=1.2, label='SMA 50')
    ax.plot(ma_200.index, ma_200, linewidth=1.2, label='SMA 200')

    ax.legend(frameon=False, loc='upper left')
    ax.set_title(f'{ticker} — Simple Moving Averages')
    ax.set_ylabel('Price')
    return fig

def plot_upward_downward_runs(data, ticker):
    prices = data['Close']
    ma_50 = prices.rolling(window=50).mean()
    trend = prices > ma_50

    plt.figure(figsize=(14, 6))
    plt.plot(prices.index, prices, 'k-', linewidth=1.5, label='Closing Price')
    plt.plot(prices.index, ma_50, 'b-', linewidth=2, alpha=0.7, label='50-Day MA')
    plt.fill_between(prices.index, prices, ma_50, where=trend, alpha=0.3, color='green', label='Upward Trend')
    plt.fill_between(prices.index, prices, ma_50, where=~trend, alpha=0.3, color='red', label='Downward Trend')
    plt.title(f'Upward & Downward Runs for {ticker}', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    fig = plt.gcf()
    plt.close(fig)
    return fig

def daily_returns(data):
    #SAMPLE CODES FROM GPT, PLS REPLACE WITH UR CODES
    returns = data['Close'].pct_change()
    plt.figure(figsize=(12, 6))
    plt.plot(returns.index, returns, label="Daily Returns", color="purple")
    plt.title("Daily Returns")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig = plt.gcf()
    plt.close(fig)
    return fig

def max_profit(data):
    #SAMPLE CODES FROM GPT, PLS REPLACE WITH UR CODES
    prices = data['Close']
    min_price = prices.min()
    max_price = prices.max()
    min_date = prices.idxmin()
    max_date = prices.idxmax()

    plt.figure(figsize=(12, 6))
    plt.plot(prices.index, prices, label="Closing Price", color="blue")
    plt.scatter(min_date, min_price, color="red", label=f"Buy ({min_date.date()})")
    plt.scatter(max_date, max_price, color="green", label=f"Sell ({max_date.date()})")
    plt.title("Max Profit Opportunity")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig = plt.gcf()
    plt.close(fig)
    return fig

# ---------------------------
# Streamlit App Layout
# ---------------------------
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.title("📊 Stock Analysis Dashboard")
st.write("Analyze stock performance with advanced metrics and visualizations")

# Step 1: Select Ticker
ticker_choices = ["-- Select --", "AAPL", "TSLA", "GOOGL", "AMZN", "META", "Others"]
ticker_selection = st.selectbox(
    "Select a Company Ticker",
    ticker_choices,
    index=None,   # 👈 makes it start empty
    placeholder="Choose a ticker..."  # 👈 nice hint text
)

ticker_input = ticker_selection
if ticker_selection == "Others":
    ticker_input = st.text_input("Type any ticker you want")

ticker = ticker_input.strip().upper() if ticker_input else None

# Step 2: Validate Ticker Immediately
data = None
if ticker:
    try:
        data = download_stock_data(ticker)
        if data.empty:
            st.error(f"No data found for ticker '{ticker}'. Please check and try again.")
            ticker = None   # invalidate ticker
        else:
            st.success(f"Valid ticker '{ticker}' found! ✅")
    except Exception as e:
        st.error(f"Error retrieving data for '{ticker}': {e}")
        ticker = None

# Step 3: Show Analysis Options only if ticker is valid
if ticker and data is not None and not data.empty:
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Simple Moving Average", "Upwards and Downwards Run", "Daily Returns", "Max Profit Calculations"]
    )

     # If user chooses SMA, show a secondary dropdown for window size
    
    if analysis_type == "Simple Moving Average":
        sma_window = st.selectbox(
            "Select SMA Window (days)",
            [12, 50, 200],
            index=1  # default to 50 days, optional
        )

    if st.button("Generate Analysis"):
        st.subheader("Analysis Results")
        st.write("Shows the selected analysis compared to the actual stock price")

        if analysis_type == "Simple Moving Average":
            fig = simple_moving_average(ticker, period='3y')
        elif analysis_type == "Upwards and Downwards Run":
            fig = plot_upward_downward_runs(data, ticker)
        elif analysis_type == "Daily Returns":
            fig = daily_returns(data)
        else:
            fig = max_profit(data)

        st.pyplot(fig)
else:
    if ticker_selection != "Others":
        st.info("Select a company ticker first to see analysis options.")
