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
# Core calcs
# ---------------------------
def _compute_smas(prices, windows=(12, 50, 200)):
    """O(n) sliding-window SMAs using cumulative sum."""
    s = prices.astype("float64")
    csum = s.cumsum()
    out = {}
    for w in windows:
        sma = (csum - csum.shift(w)) / w
        if w > 1:
            sma.iloc[:w-1] = s.expanding(min_periods=1).mean().iloc[:w-1]
        out[w] = sma
    return out
# ---------------------------
# Helpers
# ---------------------------
# ===== Project-1 metrics helpers =====
def run_stats_from_close(prices: pd.Series) -> dict:
    """
    Up/Down runs from close->close diffs.
    Returns: num_up_runs, num_down_runs, longest_up_streak, longest_down_streak
    """
    diff = prices.diff().fillna(0.0)
    up_mask = diff.gt(0)
    dn_mask = diff.lt(0)

    def _num_runs(mask: pd.Series) -> int:
        groups = mask.ne(mask.shift()).cumsum()
        return int(sum(mask.groupby(groups).any()))

    def _longest(mask: pd.Series) -> int:
        groups = mask.ne(mask.shift()).cumsum()
        return int((mask.groupby(groups).sum().max() or 0))

    return {
        "num_up_runs": _num_runs(up_mask),
        "num_down_runs": _num_runs(dn_mask),
        "longest_up_streak": _longest(up_mask),
        "longest_down_streak": _longest(dn_mask),
    }

def max_profit_btbs2(prices: pd.Series) -> float:
    """Best Time to Buy & Sell Stock II profit: sum of all positive day-to-day gains."""
    p = prices.dropna().to_numpy()
    prof = 0.0
    for i in range(1, len(p)):
        inc = p[i] - p[i-1]
        if inc > 0:
            prof += inc
    return float(prof)

# --- MACD helpers for overlays (if you want local calc) ---
def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def _macd_series(prices, fast=12, slow=26, signal=9):
    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)
    macd = ema_fast - ema_slow
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist

def _auto_rangebreaks(index):
    """Skip weekends; if intraday, also skip overnight hours."""
    breaks = [dict(bounds=["sat", "mon"])]
    try:
        diffs = pd.Series(index).diff().dropna()
        if pd.to_timedelta(diffs.median()).total_seconds() < 12 * 3600:
            breaks.append(dict(bounds=[16, 9.5], pattern="hour"))  # ~4pm–9:30am
    except Exception:
        pass
    return breaks

# ---------------------------
# Analysis Functions
# ---------------------------

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
def plotly_sma_zoom(
    data, ticker, window=50,
    show_runs=False, show_returns=False, show_maxprofit=False,
    show_momentum=False  # 👈 new
):
    prices = data['Close']
    volume = data['Volume'] if 'Volume' in data.columns else None
    ma = _compute_smas(prices, windows=(window,))[window]

    # momentum (MACD)
    if show_momentum:
        macd, macd_sig, macd_hist = _macd_series(prices)
    else:
        macd = macd_sig = macd_hist = None
    # volume colors
    up = prices.diff().fillna(0) >= 0
    vol_colors = np.where(up, "#26a69a", "#ef5350") if volume is not None else None

    # 👇 two rows: row 1 (momentum), row 2 (price + SMA + volume)
    # If momentum is off, make row1 tiny so it looks like only one chart.
    row_heights = [0.32, 0.68] if show_momentum else [0.0001, 0.9999]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=row_heights,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]  # secondary y for volume
    )

    # ----- Row 1: Momentum (MACD) -----
    if show_momentum:
        # histogram first so lines are on top
        fig.add_trace(
            go.Bar(x=macd_hist.index, y=macd_hist, name="MACD Hist",
                   marker=dict(color=np.where(macd_hist >= 0, "rgba(55, 170, 70, 0.6)", "rgba(230, 80, 80, 0.6)")),
                   opacity=0.8),
            row=1, col=1
        )
        fig.add_trace(go.Scatter(x=macd.index, y=macd, name="MACD", mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=macd_sig.index, y=macd_sig, name="Signal", mode="lines"), row=1, col=1)
        # zero reference line
        fig.add_hline(y=0, line_width=1, line_dash="dot", opacity=0.4, row=1, col=1)

    # ----- Row 2: Price + SMA (lines) -----
    fig.add_trace(go.Scatter(x=prices.index, y=prices, name="Close", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=ma.index, y=ma, name=f"SMA {window}", mode="lines"), row=2, col=1)

    # Optional overlays on price (unchanged from your version)
    if show_runs:
        ma50 = _compute_smas(prices, windows=(50,))[50]
        trend = (prices > ma50)
        up_fill = prices.where(trend)
        dn_fill = prices.where(~trend)
        # baseline + filled areas
        fig.add_trace(go.Scatter(x=ma50.index, y=ma50, mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=prices.index, y=up_fill, name="Upward Trend",
                                 mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(76,175,80,0.25)"),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=ma50.index, y=ma50, mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=prices.index, y=dn_fill, name="Downward Trend",
                                 mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(239,83,80,0.25)"),
                      row=2, col=1)

    # Volume on secondary y (row 2)
    if volume is not None:
        fig.add_trace(
            go.Bar(x=volume.index, y=volume, name="Volume",
                   marker=dict(color=vol_colors), opacity=0.7),
            row=2, col=1, secondary_y=True
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1, secondary_y=True, showgrid=False)

    # 🔧 make time scale continuous (candles “expand” as you zoom)
    rb = _auto_rangebreaks(data.index)
    fig.update_xaxes(rangebreaks=rb)   # applies to all subplots

    # Controls (attach slider/selectors to row 2/x)
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(buttons=[
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=3, label="3y", step="year", stepmode="backward"),
            dict(step="all", label="All")
        ]),
        row=2, col=1  # ← on the lower panel
    )

    fig.update_layout(
        hovermode="x unified", dragmode="pan", showlegend=True,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"{ticker} — Zoomable Price / SMA ({window}-day)"
    )
    if show_momentum:
        fig.update_yaxes(title_text="MACD", row=1, col=1, zeroline=True, zerolinecolor="rgba(255,255,255,0.15)")
    fig.update_yaxes(title_text="Price", row=2, col=1, secondary_y=False)

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

# ---------------------------
# Attempt to combine all into one Plotly chart
# ---------------------------
# ---------------------------
# Combined Plotly chart (checkbox-driven)
# ---------------------------
def plotly_combined_chart(
    data, ticker, *,
    chart_type="Line",        # "Line" | "Candlestick" | "Bars"
    sma_window=50,
    show_sma=True,
    show_runs=False,          # background green/red spans based on close-to-close
    show_maxprofit=False,
    show_returns=False,       # bottom panel bars
    show_macd=False,          # top panel
    show_rsi=False            # top panel (secondary y)
):
    prices = data["Close"].astype("float64")
    volume = data["Volume"] if "Volume" in data.columns else None

    # Indicators needed
    ma = _compute_smas(prices, windows=(sma_window,))[sma_window] if show_sma else None

    macd_line = signal_line = hist = None
    if show_macd:
        _, macd_line, signal_line, hist = Python_Project.macd_calculations(ticker)

    rsi_series = None
    if show_rsi:
        _, rsi_series = Python_Project.rsi_calculation(ticker)

    # Layout rows: [top panel?], price panel, [bottom panel?]
    top_on   = bool(show_macd or show_rsi)
    vol_on   = bool(volume is not None or show_returns)

    if top_on and vol_on:
        rows, row_heights = 3, [0.30, 0.60, 0.22]
    elif top_on and not vol_on:
        rows, row_heights = 2, [0.34, 0.66]
    elif (not top_on) and vol_on:
        rows, row_heights = 2, [0.78, 0.22]
    else:
        rows, row_heights = 1, [1.0]

    price_row  = 2 if top_on else 1
    bottom_row = rows if vol_on else price_row

    specs = []
    if rows >= 1: specs.append([{"secondary_y": True}])   # top or price
    if rows >= 2: specs.append([{"secondary_y": True}])   # price
    if rows >= 3: specs.append([{"secondary_y": True}])   # bottom (vol primary, returns secondary)

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=row_heights, specs=specs
    )

    # ----- Row 1: MACD / RSI (optional) -----
    if top_on:
        if show_macd:
            fig.add_trace(
                go.Bar(x=hist.index, y=hist, name="MACD Hist",
                       marker=dict(color=np.where(hist >= 0, "rgba(55,170,70,0.6)", "rgba(230,80,80,0.6)")),
                       opacity=0.9),
                row=1, col=1, secondary_y=False
            )
            fig.add_trace(go.Scatter(x=macd_line.index,  y=macd_line,  name="MACD",   mode="lines"),
                          row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=signal_line.index, y=signal_line, name="Signal", mode="lines"),
                          row=1, col=1, secondary_y=False)
            fig.add_hline(y=0, line_width=1, line_dash="dot", opacity=0.4, row=1, col=1)

        if show_rsi:
            fig.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series, name="RSI (14)", mode="lines"),
                          row=1, col=1, secondary_y=True)
            fig.add_hrect(y0=30, y1=70, line_width=0, fillcolor="rgba(200,200,255,0.12)",
                          row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=1, col=1, secondary_y=True)

        if show_macd:
            fig.update_yaxes(title_text="MACD", row=1, col=1, secondary_y=False)

    # ----- Row 2: Price panel (Line / Candles / Bars) -----
    has_ohlc = {"Open", "High", "Low", "Close"}.issubset(data.columns)
    if chart_type == "Candlestick" and has_ohlc:
        fig.add_trace(go.Candlestick(
            x=data.index, open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"], name="Candles",
            increasing_line_width=2.2, decreasing_line_width=2.2,  # thicker bodies
            whiskerwidth=0.9  # thicker wicks
        ), row=price_row, col=1, secondary_y=False)
        fig.update_traces(
            selector=dict(type="candlestick"),
            increasing_fillcolor="rgba(46,204,113,1.0)",
            decreasing_fillcolor="rgba(231,76,60,1.0)",
            increasing_line_color="rgba(46,204,113,1.0)",
            decreasing_line_color="rgba(231,76,60,1.0)"
        )
    elif chart_type == "Bars" and has_ohlc:
        fig.add_trace(go.Ohlc(
            x=data.index, open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"], name="OHLC",
            increasing_line_width=2.2, decreasing_line_width=2.2
        ), row=price_row, col=1, secondary_y=False)
    else:
        fig.add_trace(go.Scatter(x=prices.index, y=prices, name="Close", mode="lines"),
                      row=price_row, col=1, secondary_y=False)

    if show_sma and ma is not None:
        fig.add_trace(go.Scatter(x=ma.index, y=ma, name=f"SMA {sma_window}", mode="lines", line=dict(width=2.2)),
                      row=price_row, col=1, secondary_y=False)

    # Up/Down runs background spans (close→close)
    if show_runs:
        diff = prices.diff().fillna(0.0)
        sign = np.sign(diff)
        sign = sign.replace(0, method="ffill").fillna(0)
        start = None
        current = None
        for i, (ts, s) in enumerate(sign.items()):
            if current is None:
                current, start = s, ts
                continue
            if s != current:
                end = sign.index[i-1]
                if current != 0:
                    fig.add_vrect(x0=start, x1=end,
                                  fillcolor="rgba(76,175,80,0.10)" if current > 0 else "rgba(239,83,80,0.10)",
                                  line_width=0, layer="below", row=price_row, col=1)
                start, current = ts, s
        if start is not None and current != 0:
            fig.add_vrect(x0=start, x1=sign.index[-1],
                          fillcolor="rgba(76,175,80,0.10)" if current > 0 else "rgba(239,83,80,0.10)",
                          line_width=0, layer="below", row=price_row, col=1)

    # Max-profit markers (guarded)
    if show_maxprofit:
        try:
            trades_df, buy_days, f_buy_days, f_sell_days, f_buy_px, f_sell_px, _ = Python_Project.max_profit_calculations(ticker)
            if f_buy_days and f_sell_days:
                fig.add_trace(go.Scatter(x=f_buy_days, y=f_buy_px, mode="markers",
                                         name="Buy",  marker=dict(symbol="triangle-up",   size=12, color="#2e7d32")),
                              row=price_row, col=1)
                fig.add_trace(go.Scatter(x=f_sell_days, y=f_sell_px, mode="markers",
                                         name="Sell", marker=dict(symbol="triangle-down", size=12, color="#c62828")),
                              row=price_row, col=1)
        except Exception as e:
            st.warning(f"Max Profit couldn’t run: {e}")

    # ----- Bottom panel: Volume (primary) + Returns (secondary) -----
    if vol_on:
        if volume is not None:
            up = prices.diff().fillna(0) >= 0
            vol_colors = np.where(up, "#26a69a", "#ef5350")
            fig.add_trace(
                go.Bar(x=volume.index, y=volume, name="Volume",
                       marker=dict(color=vol_colors), opacity=0.8),
                row=bottom_row, col=1, secondary_y=False
            )
            fig.update_yaxes(title_text="Volume", row=bottom_row, col=1, secondary_y=False, showgrid=False)
        if show_returns:
            rets = prices.pct_change() * 100.0
            fig.add_trace(
                go.Bar(x=rets.index, y=rets, name="Daily Returns (%)", opacity=0.75),
                row=bottom_row, col=1, secondary_y=True
            )
            fig.update_yaxes(title_text="Returns (%)", row=bottom_row, col=1, secondary_y=True)

    # ----- Axes / Interactions -----
    # One simple bottom bar: turn OFF Plotly rangeslider everywhere
    for r in range(1, rows + 1):
        fig.update_xaxes(rangeslider_visible=False, row=r, col=1)

    # Range buttons on price row
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ]
        ),
        row=price_row, col=1
    )

    # Skip weekends/overnights so candles expand as you zoom
    rb = _auto_rangebreaks(data.index)
    for r in range(1, rows + 1):
        fig.update_xaxes(rangebreaks=rb, row=r, col=1)

    fig.update_layout(
        hovermode="x unified", dragmode="pan", showlegend=True,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"{ticker} — {chart_type} + Overlays"
    )
    fig.update_yaxes(title_text="Price", row=price_row, col=1, secondary_y=False)
    fig.update_yaxes(zeroline=False, row=price_row, col=1)
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")
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
    # ---------- Checkbox-only controls (no dropdown) ----------
# 1) Base chart controls
    chart_type = st.selectbox("Chart Type", ["Line", "Candlestick", "Bars"])
    sma_window = st.slider("SMA Window (days)", 5, 250, 50, 1)
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Simple Moving Average", "Upwards and Downwards Run", "Daily Returns", "Max Profit Calculations", "MACD (Moving Average Convergence Diverence)", "RSI (Relative Strength Index)"]
    )

    if analysis_type == "Simple Moving Average":
    # --- Top controls ---
        chart_type = st.selectbox("Chart Type", ["Line", "Candlestick", "Bars"])
        sma_window = st.slider("SMA Window (days)", 5, 250, 50, 1)

    # 2) Feature toggles (layer onto the SAME Plotly chart)
    c1, c2, c3 = st.columns(3)
    with c1:
        cb_sma  = st.checkbox("SMA", value=True)
        cb_runs = st.checkbox("Trend Runs", value=False)
    with c2:
        cb_macd = st.checkbox("MACD (top panel)", value=False)
        cb_rsi  = st.checkbox("RSI (top panel)", value=False)
    with c3:
        cb_rets = st.checkbox("Daily Returns (bottom)", value=False)
        cb_maxp = st.checkbox("Max Profit markers", value=False)

    # combined Plotly chart
    fig = plotly_combined_chart(
        data, ticker,
        chart_type=chart_type,
        sma_window=sma_window,
        show_sma=cb_sma,
        show_runs=cb_runs,
        show_maxprofit=cb_maxp,
        show_returns=cb_rets,
        show_macd=cb_macd,   # ← MACD in a separate graph ABOVE
        show_rsi=cb_rsi      # ← RSI can share that top panel
    )
    # render last
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    # 4) Show explanations ONLY for enabled features (readable in light/dark)
    exp = []
    if cb_sma:   exp.append("**SMA**: Average closing price over *N* days. Smooths noise and shows trend.")
    if cb_runs:  exp.append("**Up/Down Runs**: Background is 🟢 when price rises day-to-day, 🔴 when it falls. Longer stretches = stronger trend.")
    if cb_macd:  exp.append("**MACD**: Momentum. Line = 12–26 EMA gap, Signal = 9-EMA of that, bars show the gap size.")
    if cb_rsi:   exp.append("**RSI**: 0–100 momentum scale; >70 often overbought, <30 oversold.")
    if cb_rets:  exp.append("**Daily Returns**: % change from previous close, shown as bars below.")
    if cb_maxp:  exp.append("**Max Profit**: Illustrative best ▲ buys and ▼ sells from your helper function.")

    if exp:
        st.info("**Feature Explanations**\n\n- " + "\n- ".join(exp))
    with st.expander("📊 Project-1 Metrics", expanded=True):
        prices = data["Close"].astype("float64")

        sma_last = _compute_smas(prices, windows=(sma_window,))[sma_window].iloc[-1] if cb_sma else prices.iloc[-1]
        st.write(f"**SMA {sma_window} (last)**: {sma_last:,.2f}")

        rs = run_stats_from_close(prices)
        st.write("**Runs (Close→Close)**")
        st.write(
            f"- Up runs: **{rs['num_up_runs']}**, Down runs: **{rs['num_down_runs']}**  \n"
            f"- Longest up streak: **{rs['longest_up_streak']} days**, "
            f"Longest down streak: **{rs['longest_down_streak']} days**"
        )

        last_ret = prices.pct_change().iloc[-1] * 100.0
        st.write(f"**Daily Return (last)**: {last_ret:,.2f}%")

        btbs2 = max_profit_btbs2(prices)
        st.write(f"**Max Profit (BTBS-II)**: ${btbs2:,.2f}")

else:
    if ticker_selection != "Others":
        st.info("Select a company ticker first to see analysis options.")

# ---------------------------
# Previous dropdown-based UI (disabled / stored for if needed)
# ---------------------------
'''
    # --- Previous UI ---
    if False:
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Simple Moving Average", "Upwards and Downwards Run",
            "Daily Returns", "Max Profit Calculations", "MACD", "RSI"]
        )

        fig = None  # avoid linter warnings

        if analysis_type == "Simple Moving Average":
            # --- Top controls ---
            chart_type = st.selectbox("Chart Type", ["Line", "Candlestick", "Bars"])
            sma_window = st.slider("SMA Window (days)", 5, 250, 50, 1)

            # --- Feature checkboxes (layer onto SAME chart) ---
            c1, c2, c3 = st.columns(3)
            with c1:
                cb_sma  = st.checkbox("SMA", value=True)
                cb_runs = st.checkbox("Trend Runs", value=False)
            with c2:
                cb_macd = st.checkbox("MACD (top panel)", value=False)
                cb_rets = st.checkbox("Daily Returns (bottom)", value=False)
            with c3:
                cb_maxp = st.checkbox("Max Profit markers", value=False)
                cb_rsi  = st.checkbox("RSI (top panel)", value=False)

            fig = plotly_combined_chart(
                data, ticker,
                chart_type=chart_type,
                sma_window=sma_window,
                show_sma=cb_sma,
                show_runs=cb_runs,
                show_maxprofit=cb_maxp,
                show_returns=cb_rets,
                show_macd=cb_macd,
                show_rsi=cb_rsi
            )
            st.plotly_chart(fig, use_container_width=True)

            # explanations only when enabled
            exp = []
            if cb_sma:   exp.append("**SMA**: Average price over N days to smooth noise.")
            if cb_runs:  exp.append("**Trend Runs**: Green when price > SMA50, red otherwise; long runs suggest stronger trends.")
            if cb_macd:  exp.append("**MACD**: 12/26 EMA spread with 9-EMA signal; histogram above/below 0 shows momentum bias.")
            if cb_rsi:   exp.append("**RSI**: 0–100 scale; >70 overbought, <30 oversold (guide band shown).")
            if cb_rets:  exp.append("**Daily Returns**: % change each day (bottom secondary axis).")
            if cb_maxp:  exp.append("**Max Profit**: Illustrative best buy/sell points in the window (▲ buy, ▼ sell).")

            if exp:
                st.info("**Feature Explanations**\n\n- " + "\n- ".join(exp))

        if st.button("Generate Analysis"):
            st.subheader("Analysis Results")
            st.write("Shows the selected analysis compared to the actual stock price")

            if analysis_type == "Max Profit Calculations":
                fig, trades_df = max_profit(ticker)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                if trades_df is not None:
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
                elif analysis_type == "RSI":
                    fig = rsi(ticker)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
    '''
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
                st.pyplot(fig)
                plt.close(fig)
                #st.plotly_chart(fig, use_container_width=True)

else:
    if ticker_selection != "Others":
        st.info("Select a company ticker first to see analysis options.")
