from plotly.subplots import make_subplots
import plotly.graph_objects as go
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
    """O(n) sliding-window SMAs using one cumulative sum pass."""
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
    returns = data['Close'].pct_change()
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#e6e6e6')
    ax.set_facecolor('#e6e6e6')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.tick_params(axis='y', right=True, left=False, colors='#222')
    ax.tick_params(axis='x', colors='#222')

    ax.plot(returns.index, returns, label="Daily Returns", color="purple", linewidth=1.2)
    ax.set_title("Daily Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.legend(frameon=False, loc='upper left')
    fig.autofmt_xdate()
    plt.close(fig)
    return fig

def max_profit(ticker):
    try:
        # Call the function from Python_Project.py
        trades_df, buy_days, filtered_buy_days, filtered_sell_days, filtered_buy_prices, filtered_sell_prices, total_profit = Python_Project.max_profit_calculations(ticker)
        
        if not buy_days:
            st.warning("No profitable trades found in this period.")
            return None
        else:
            st.success(f"💰 Total Potential Profit: ${total_profit:.2f}")
    
            # Plotting
            data = download_stock_data(ticker)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data['Close'], label='Close Price', color='blue')
            ax.scatter(filtered_buy_days, filtered_buy_prices, marker='^', color='green', label='Buy', s=50, alpha=0.7)
            ax.scatter(filtered_sell_days, filtered_sell_prices, marker='v', color='red', label='Sell', s=50, alpha=0.7)
            ax.set_title(f"{ticker} Price with Top Max Profit Trades")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.legend()
            ax.grid(True)

            return fig,trades_df
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")

        
def plotly_sma_zoom(data, ticker, window=50):
    prices = data['Close']
    volume = data['Volume'] if 'Volume' in data.columns else None
    ma = _compute_smas(prices, windows=(window,))[window]

    up = prices.diff().fillna(0) >= 0
    vol_colors = np.where(up, "#26a69a", "#ef5350") if volume is not None else None

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.72, 0.28], specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    fig.add_trace(go.Scatter(x=prices.index, y=prices, name="Close", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ma.index, y=ma, name=f"SMA {window}", mode="lines"), row=1, col=1)
    if volume is not None:
        fig.add_trace(go.Bar(x=volume.index, y=volume, name="Volume",
                             marker=dict(color=vol_colors), opacity=0.7), row=2, col=1)
    fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(buttons=[
        dict(count=1, label="1m", step="month", stepmode="backward"),
        dict(count=3, label="3m", step="month", stepmode="backward"),
        dict(count=6, label="6m", step="month", stepmode="backward"),
        # 🔧 FIX: replace 'yeartodate' with step='year', stepmode='todate'
        dict(label="YTD", step="year", stepmode="todate"),
        dict(count=1, label="1y", step="year", stepmode="backward"),
        dict(count=3, label="3y", step="year", stepmode="backward"),
        dict(step="all", label="All")
    ]),
    row=1, col=1
)
    fig.update_layout(
        hovermode="x unified", dragmode="pan", showlegend=True,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"{ticker} — Zoomable Price / SMA ({window}-day)"
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1, showgrid=False)
    return fig



def macd(ticker):
    prices, macd_line, signal_line, histogram = Python_Project.macd_calculations(ticker)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1.5]})

    # Plotting the price chart with MACD overlay
    ax1.plot(prices.index, prices, label='Close Price', color='blue', linewidth=1.5)
    ax1.set_ylabel('Price ($)', color='black')
    ax1.tick_params(axis='x', labelcolor='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Date')
    ax3 = ax1.twinx()
    ax3.plot(prices.index, macd_line, label='MACD Line', color='purple', linewidth=1)
    ax3.plot(prices.index, signal_line, label='Signal Line', color='orange', linewidth=1)
    ax3.set_ylabel('MACD', color='purple')
    ax3.tick_params(axis='y', labelcolor='purple')
    ax3.legend(loc='upper left')
    
    # Secondary chart for histogram comparison
    ax2.plot(prices.index, macd_line, label= 'MACD Line', color='purple')
    ax2.plot(prices.index, signal_line, label= 'Signal Line', color='orange')
    colors = ['green' if h >= 0 else 'red' for h in histogram]
    ax2.bar(prices.index, histogram, color=colors, alpha=0.5)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Histogram')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')


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
        ["Simple Moving Average", "Upwards and Downwards Run", "Daily Returns", "Max Profit Calculations", "MACD"]
    )

    if analysis_type == "Simple Moving Average":
        # ✅ define the slider BEFORE using sma_window
        sma_window = st.slider("SMA Window (days)", min_value=5, max_value=250, value=50, step=1)
        # live interactive chart
        st.plotly_chart(plotly_sma_zoom(data, ticker, window=sma_window), use_container_width=True)

    if st.button("Generate Analysis"):
        st.subheader("Analysis Results")
        st.write("Shows the selected analysis compared to the actual stock price")

        if analysis_type == "Max Profit Calculations":
            fig,trades_df = max_profit(ticker)
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            if trades_df is not None:
                # Display trades in an expandable section, user can sort columns too
                with st.expander("📊 View Trade Details"):
                    st.dataframe(trades_df.style.format({
                        "Buy Price": "{:.2f}",
                        "Sell Price": "{:.2f}", 
                        "Profit": "{:.2f}"
                    }), use_container_width=True)
        else:
            if analysis_type == "Simple Moving Average":
                st.plotly_chart(plotly_sma_zoom(data, ticker, window=sma_window), use_container_width=True)
            elif analysis_type == "Upwards and Downwards Run":
                fig = plot_upward_downward_runs(data, ticker)
            elif analysis_type == "Daily Returns":
                fig = daily_returns(data)
            elif analysis_type == "MACD":
                fig = macd(ticker)
            if fig:
                st.pyplot(fig)
                plt.close(fig)
        

else:
    if ticker_selection != "Others":
        st.info("Select a company ticker first to see analysis options.")
