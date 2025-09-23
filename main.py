import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import Python_Project
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
            sma.iloc[:w-1] = s.expanding(min_periods=1).mean().iloc[:w-1]
        out[w] = sma
    return out

def simple_moving_average(ticker, window=50, period='3y'):
    data = download_stock_data(ticker, period)
    prices = data['Close']
     # Compute only the selected window
    smas = _compute_smas(prices, windows=(window,))
    ma = smas[window]

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
    ax.plot(ma.index, ma, linewidth=1.2, label=f'SMA {window}')

    ax.legend(frameon=False, loc='upper left')
    ax.set_title(f'{ticker} — Simple Moving Average ({window}-day)')
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

def max_profit(ticker):
    try:
        trades, buy_days, sell_days, buy_prices, sell_prices, total_profit = Python_Project.plot_max_profit_calculations(ticker)
        
        if not buy_days:
            st.warning("No profitable trades found in this period.")
        else:
            st.success(f"Total Potential Profit: ${total_profit}")

            # Plotting
            data = download_stock_data(ticker)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data['Close'], label='Close Price', color='blue')
            ax.scatter(buy_days, buy_prices, marker='^', color='green', label='Buy', s=100)
            ax.scatter(sell_days, sell_prices, marker='v', color='red', label='Sell', s=100)
            ax.set_title(f"{ticker} Price with Max Profit Trades")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig, clear_figure=True)

            with st.expander("📊 View Trade Details"):
                st.dataframe(trades, use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching data: {e}")
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
            fig = simple_moving_average(ticker, period='3y', window=sma_window)
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
