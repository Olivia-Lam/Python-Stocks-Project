from plotly.subplots import make_subplots
import plotly.graph_objects as go
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

# Max Profit Calculation Function
def max_profit_calculations(ticker, period = '3y'):
    data = download_stock_data(ticker, period)
    prices = data['Close']
    buy_days = []
    sell_days = []
    prices_bought = []
    prices_sold = []
    trades = []
    filtered_buy_days = []
    filtered_sell_days = []
    filtered_buy_prices = []
    filtered_sell_prices = []
    total_profit = 0
    i = 0
    n = len(prices)
    price_list = prices.tolist()
    dates = prices.index.tolist()
    
    # Loop to buy low and sell high
    while i < n - 1:
        # Find buy point, keeps moving until it detects a rise
        while i < n - 1 and price_list[i+1] <= price_list[i]:
            i += 1
        if i == n - 1:
            break
        #Once the greatest rise is detected, that will be stored as a buy point
        buy_price = price_list[i]
        buy_day = dates[i]
        buy_days.append(buy_day)
        prices_bought.append(buy_price)
        
        # Find sell point, reverse of buy point
        while i < n - 1 and price_list[i+1] >= price_list[i]:
            i += 1
        # Once the highest point is detected, that will be stored as a sell point
        sell_price = price_list[i]
        sell_day = dates[i]
        sell_days.append(sell_day)
        prices_sold.append(sell_price)
        
        # Calculate profit from the transaction
        profit = sell_price - buy_price
        # Sums up all the profits from each transaction
        total_profit += profit

        # Record the trade to put into table later on
        trades.append({
            "Buy Date": buy_day.strftime("%Y-%m-%d"),
            "Buy Price": round(buy_price,2),
            "Sell Date": sell_day.strftime("%Y-%m-%d"),
            "Sell Price": round(sell_price,2),
            "Profit": round(profit,2)
        }
        )
        # Since there are so many transactions, we want to filter for the significant ones to display (Top 10% in this case)
        trades_df = pd.DataFrame(trades)
        top_trades = trades_df["Profit"].quantile(0.9)
        # Stores trades with profits above 90th percentile
        if profit >= top_trades:
            filtered_buy_days.append(buy_day)
            filtered_sell_days.append(sell_day)
            filtered_buy_prices.append(buy_price)
            filtered_sell_prices.append(sell_price)
    
    return trades_df, buy_days, filtered_buy_days, filtered_sell_days, filtered_buy_prices, filtered_sell_prices, round(total_profit, 2)

# Max Profit Indicator
def max_profit(ticker):
    try:
        
        trades_df, buy_days, filtered_buy_days, filtered_sell_days, filtered_buy_prices, filtered_sell_prices, total_profit = max_profit_calculations(ticker)
        
        if not buy_days:
            st.warning("No profitable trades found in this period.")
            return None
        else:
            st.success(f"💰 Total Potential Profit: ${total_profit:.2f}")

        
            fig = go.Figure()

            # Add closing price line
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue')
            ))

            # Add buy points
            fig.add_trace(go.Scatter(
                x=filtered_buy_days,
                y=filtered_buy_prices,
                mode='markers',
                marker=dict(symbol='triangle-up', color='green', size=10),
                name='Buy'
            ))

            # Add sell points
            fig.add_trace(go.Scatter(
                x=filtered_sell_days,
                y=filtered_sell_prices,
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=10),
                name='Sell'
            ))

            # Update layout for interactivity
            fig.update_layout(
                title=f"{ticker} Price with Most Profitable Trades",
                xaxis_title='Date',
                yaxis_title='Price ($)',
                xaxis_rangeslider_visible=True,  # adds a zoom slider at bottom
                hovermode='x unified',
                template='plotly_white'
            )

            return fig,trades_df

    except Exception as e:
        st.error(f"Error fetching data: {e}")

#Zoom Function
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

# MACD Calculation Function
def macd_calculations(ticker, period = '3y'):
    data = download_stock_data(ticker, period)
    prices = data['Close']

    # Exonential Moving Averages calculation
    ema_12 = prices.ewm(span=12, adjust=False).mean()
    ema_26 = prices.ewm(span=26, adjust=False).mean()

    # MACD and signal line calculation
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    return prices, macd_line, signal_line, histogram

# MACD Indicator 
def macd(ticker):
    try:

        prices, macd_line, signal_line, histogram = macd_calculations(ticker)
        
        # For histogram scaling and colors
        scale_factor = 2
        scaled_histogram = histogram * scale_factor
        colors = ['green' if h >= 0 else 'red' for h in scaled_histogram]

        # Create subplots: 2 rows, shared x-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.6, 0.4],
            vertical_spacing=0.2,
            subplot_titles=(f"{ticker} Price Chart", "MACD Histogram")
        )

        # Price chart display
        fig.add_trace(
            go.Scatter(x=prices.index, y=prices, mode='lines', name='Close Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        # MACD with Histogram
        fig.add_trace(
            go.Bar(x=prices.index, y=scaled_histogram, name='Histogram', marker_color=colors, opacity=0.5),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=prices.index, y=macd_line, mode='lines', name='MACD Line', line=dict(color='purple')),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=prices.index, y=signal_line, mode='lines', name='Signal Line', line=dict(color='orange')),
            row=2, col=1
        )

        # Adding white line for MACD indicator zero line
        fig.add_shape(
            type="line",
            x0=prices.index[0], x1=prices.index[-1],
            y0=0, y1=0,
            line=dict(color="white", width=1, dash="dash"),
            row=2, col=1
        )

        # Layout and interactive features
        fig.update_layout(
            height=700,
            hovermode='x unified',
            template='plotly_white',
            xaxis2_rangeslider_visible=True,  # interactive range slider for histogram
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig
    
    except Exception as e:
        st.error(f"Error fetching data: {e}")

# RSI Calculation Function
def rsi_calculation(ticker, period='3y', window=14):
    data = download_stock_data(ticker, period)
    prices = data['Close']

    # Calculate daily price changes
    difference = prices.diff()

    # Separate gains and losses
    gain = difference.where(difference > 0, 0)
    loss = -difference.where(difference < 0, 0)

    # Calculate average gain & loss
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # Calculate RS (needed for RSI)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return prices, rsi

# RSI Indicator
def rsi(ticker):
    try:

        prices, rsi = rsi_calculation(ticker)

        # Create subplots: 2 rows, shared x-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.6, 0.4],
            vertical_spacing=0.2,
            subplot_titles=(f"{ticker} Price", f"{ticker} RSI")
        )

        # Price Chart Display
        fig.add_trace(
            go.Scatter(x=prices.index, y=prices, mode='lines', name='Close Price', line=dict(color='blue')),
            row=1, col=1
        )

        # RSI Chart Display
        fig.add_trace(
            go.Scatter(x=prices.index, y=rsi, mode='lines', name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        # Adding grey line for RSI midiline
        fig.add_shape(
            type="line",
            x0=prices.index[0], x1=prices.index[-1],
            y0=50, y1=50,
            line=dict(color="grey", width=1, dash="dash"),
            row=2, col=1
        )

        # Add RSI overbought/oversold lines
        fig.add_trace(
            go.Scatter(x=prices.index, y=[70]*len(prices), mode='lines',
                    name='Overbought (70)', line=dict(color='red', dash='dash'), opacity=0.7),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=prices.index, y=[30]*len(prices), mode='lines',
                    name='Oversold (30)', line=dict(color='green', dash='dash'), opacity=0.7),
            row=2, col=1
        )
        
        # Layout and interactive features
        fig.update_layout(
            height=700,
            hovermode='x unified',
            template='plotly_white',
            xaxis2_rangeslider_visible=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig
    
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
        ["Simple Moving Average", "Upwards and Downwards Run", "Daily Returns", "Max Profit Calculations", "MACD (Moving Average Convergence Diverence)", "RSI (Relative Strength Index)"]
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
            # fig,trades_df = max_profit(ticker)
            fig,trades_df = max_profit(ticker)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
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
            elif analysis_type == "MACD (Moving Average Convergence Diverence)":
                fig = macd(ticker)
            elif analysis_type == "RSI (Relative Strength Index)":
                fig = rsi(ticker)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
        

else:
    if ticker_selection != "Others":
        st.info("Select a company ticker first to see analysis options.")
