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


# Calculate daily returns 
def daily_returns(data):
    current_price = data['Close']

    previous_price = data['Close'].shift(1)

    # Daily return formula
    daily_return = (current_price - previous_price) / previous_price
    
    # return value as percentage
    return (daily_return * 100.0)

# Identify consecutive upward and downward runs
def identify_consecutive_runs(data):
    
    # Identify consecutive upward and downward runs based on close-to-close price changes.
    # Returns run data with statistics for visualization and analysis.
    
    prices = data['Close']
    
    # Calculate daily price changes (positive = up, negative = down)
    price_changes = prices.diff()
    
    # Create run direction: 1 for up, -1 for down, 0 for flat
    directions = np.where(price_changes > 0, 1, np.where(price_changes < 0, -1, 0))
    
    # Identify runs
    runs = []
    current_direction = None
    run_start = None
    run_length = 0
    
    for i, direction in enumerate(directions):
        if direction == 0:  # Skip flat days
            continue
            
        if current_direction != direction:
            # End previous run
            if current_direction is not None and run_length > 0:
                runs.append({
                    'direction': 'Upward' if current_direction == 1 else 'Downward',
                    'start_idx': run_start,
                    'end_idx': i - 1,
                    'length': run_length,
                    'start_date': prices.index[run_start],
                    'end_date': prices.index[i - 1],
                    'start_price': prices.iloc[run_start],
                    'end_price': prices.iloc[i - 1]
                })
            
            # Start new run
            current_direction = direction
            run_start = i
            run_length = 1
        else:
            # Continue current run
            run_length += 1
    
    # Add final run
    if current_direction is not None and run_length > 0:
        runs.append({
            'direction': 'Upward' if current_direction == 1 else 'Downward',
            'start_idx': run_start,
            'end_idx': len(directions) - 1,
            'length': run_length,
            'start_date': prices.index[run_start],
            'end_date': prices.index[-1],
            'start_price': prices.iloc[run_start],
            'end_price': prices.iloc[-1]
        })
    
    runs_df = pd.DataFrame(runs)
    
    # Calculate statistics
    if not runs_df.empty:
        upward_runs = runs_df[runs_df['direction'] == 'Upward']
        downward_runs = runs_df[runs_df['direction'] == 'Downward']
        
        stats = {
            'total_upward_runs': len(upward_runs),
            'total_downward_runs': len(downward_runs),
            'longest_upward_run': upward_runs['length'].max() if len(upward_runs) > 0 else 0,
            'longest_downward_run': downward_runs['length'].max() if len(downward_runs) > 0 else 0,
            'avg_upward_length': upward_runs['length'].mean() if len(upward_runs) > 0 else 0,
            'avg_downward_length': downward_runs['length'].mean() if len(downward_runs) > 0 else 0
        }
    else:
        stats = {
            'total_upward_runs': 0,
            'total_downward_runs': 0,
            'longest_upward_run': 0,
            'longest_downward_run': 0,
            'avg_upward_length': 0,
            'avg_downward_length': 0
        }
    
    return runs_df, stats

# Display run statistics
def display_run_statistics(stats):
    """
    Display upward and downward run statistics in Streamlit
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "📈 Upward Runs", 
            f"{stats['total_upward_runs']}", 
            f"Longest: {int(stats['longest_upward_run'])} days"
        )
        if stats['avg_upward_length'] > 0:
            st.caption(f"Average length: {stats['avg_upward_length']:.1f} days")
    
    with col2:
        st.metric(
            "📉 Downward Runs", 
            f"{stats['total_downward_runs']}", 
            f"Longest: {int(stats['longest_downward_run'])} days"
        )
        if stats['avg_downward_length'] > 0:
            st.caption(f"Average length: {stats['avg_downward_length']:.1f} days")

# Max Profit Calculation Function
def max_profit_calculations(data):
    
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
    filtered = trades_df[trades_df["Profit"] >= top_trades]
    # To use for marker plotting
    filtered_buy_days = pd.to_datetime(filtered["Buy Date"]).tolist()
    filtered_sell_days = pd.to_datetime(filtered["Sell Date"]).tolist()
    filtered_buy_prices = filtered["Buy Price"].tolist()
    filtered_sell_prices = filtered["Sell Price"].tolist()
    
    return trades_df, filtered_buy_days, filtered_sell_days, filtered_buy_prices, filtered_sell_prices, round(total_profit, 2)


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
    prices, macd_line, signal_line, histogram = macd_calculations(ticker)
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
            go.Bar(x=histogram.index, y=histogram, name="MACD Hist",
                   marker=dict(color=np.where(histogram >= 0, "rgba(55, 170, 70, 0.6)", "rgba(230, 80, 80, 0.6)")),
                   opacity=0.8),
            row=1, col=1
        )
        fig.add_trace(go.Scatter(x=macd_line.index, y=macd_line, name="MACD", mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=signal_line.index, y=signal_line, name="Signal", mode="lines"), row=1, col=1)
        # zero reference line
        fig.add_hline(y=0, line_width=1, line_dash="dot", opacity=0.4, row=1, col=1)

    # ----- Row 2: Price + SMA (lines) -----
    fig.add_trace(go.Scatter(x=prices.index, y=prices, name="Close", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=ma.index, y=ma, name=f"SMA {window}", mode="lines"), row=2, col=1)

    # Optional overlays on price (FIXED TREND RUNS LOGIC)
    if show_runs:
        ma50 = _compute_smas(prices, windows=(50,))[50]
        trend_up = (prices > ma50)  # True when price is above MA50 (upward trend)
        
        # Create filled areas for upward and downward trends
        # GREEN for upward (price > MA), RED for downward (price < MA)
        fig.add_trace(go.Scatter(
            x=prices.index, 
            y=ma50, 
            mode="lines",
            line=dict(width=0), 
            showlegend=False, 
            hoverinfo="skip",
            name="MA50_baseline"
        ), row=2, col=1)
        
        # GREEN fill for upward trend (price above MA50)
        fig.add_trace(go.Scatter(
            x=prices.index, 
            y=prices.where(trend_up), 
            name="📈 Upward Trend",
            mode="lines", 
            line=dict(width=0),
            fill="tonexty", 
            fillcolor="rgba(34, 139, 34, 0.3)",  # Forest Green with transparency
            hovertemplate="Upward Trend<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"
        ), row=2, col=1)
        
        # Add MA50 baseline again for downward fill
        fig.add_trace(go.Scatter(
            x=prices.index, 
            y=ma50, 
            mode="lines",
            line=dict(width=0), 
            showlegend=False, 
            hoverinfo="skip",
            name="MA50_baseline2"
        ), row=2, col=1)
        
        # RED fill for downward trend (price below MA50)
        fig.add_trace(go.Scatter(
            x=prices.index, 
            y=prices.where(~trend_up), 
            name="📉 Downward Trend",
            mode="lines", 
            line=dict(width=0),
            fill="tonexty", 
            fillcolor="rgba(220, 20, 60, 0.3)",  # Crimson Red with transparency
            hovertemplate="Downward Trend<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"
        ), row=2, col=1)
        
        # Add the MA50 line on top for visibility
        fig.add_trace(go.Scatter(
            x=ma50.index, 
            y=ma50, 
            name="MA50 Reference",
            mode="lines", 
            line=dict(color="#4169E1", width=1.5, dash="dot"),  # Royal Blue dotted line
            hovertemplate="MA50: $%{y:.2f}<extra></extra>"
        ), row=2, col=1)

    # Volume on secondary y (row 2)
    if volume is not None:
        fig.add_trace(
            go.Bar(x=volume.index, y=volume, name="Volume",
                   marker=dict(color=vol_colors), opacity=0.7),
            row=2, col=1, secondary_y=True
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1, secondary_y=True, showgrid=False)

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
def macd_calculations(data):
    
    prices = data['Close']

    # Obtain exponential moving average data
    ema_12 = prices.ewm(span=12, adjust=False).mean()
    ema_26 = prices.ewm(span=26, adjust=False).mean()

    # Get MACD, signal line info and obtain histogram results for plotting
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    return prices, macd_line, signal_line, histogram


# RSI Calculation Function
def rsi_calculation(data, window=14):

    prices = data['Close']

    # Calculate daily price changes
    difference = prices.diff()

    # Separate gains and losses
    gain = difference.where(difference > 0, 0)
    loss = -difference.where(difference < 0, 0)

    # Calculate average gain & loss
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # Relative strength calculation
    rs = avg_gain / avg_loss

    # Relative strength INDEX calculation
    rsi = 100 - (100 / (1 + rs))

    return prices, rsi


def plotly_combined_chart(
    data, ticker, *,
    sma_window=50,
    show_sma=True,
    show_runs=False,
    show_maxprofit=False,
    show_returns=False,
    show_macd=False,
    show_rsi=False
):
    prices = data["Close"]

    # Determine the number of rows dynamically
    num_rows = 1  # For the main price chart
    if show_macd:
        num_rows += 1
    if show_rsi:
        num_rows += 1
    if show_returns:
        num_rows += 1
    
    # Calculate row heights: indicators get 20% of space, price chart gets the rest
    indicator_heights = 0.20
    price_height = 1 - (num_rows - 1) * indicator_heights
    row_heights = [indicator_heights] * (num_rows - 1) + [price_height]
    
    # All subplots will have a single y-axis since returns and volume are now on their own charts
    specs = [[{"secondary_y": False}]] * num_rows

    # Initialize the subplot figure
    fig = make_subplots(
        rows=num_rows, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=row_heights,
        specs=specs
    )

    # Determine the row for the main price chart
    price_row = num_rows
    current_indicator_row = 0

    # --- Add MACD to the first available indicator row ---
    if show_macd:
        current_indicator_row += 1
        # Assumed function
        prices, macd_line, signal_line, histogram = macd_calculations(data)
        fig.add_trace(go.Bar(
            x=histogram.index, y=histogram, name="MACD Hist",
            marker=dict(color=["#26a69a" if h > 0 else "#ef5350" for h in histogram]),
            opacity=0.5
        ), row=current_indicator_row, col=1)
        fig.add_trace(go.Scatter(x=macd_line.index, y=macd_line, name="MACD Line", line=dict(color="#2962FF")),
                      row=current_indicator_row, col=1)
        fig.add_trace(go.Scatter(x=signal_line.index, y=signal_line, name="Signal Line", line=dict(color="#FF6D00")),
                      row=current_indicator_row, col=1)
        fig.update_yaxes(title_text="MACD", row=current_indicator_row, col=1)
        fig.add_hline(y=0, line_width=1, line_dash="dash", opacity=0.4, line_color="#888", row=current_indicator_row, col=1)

    # --- Add RSI to the next available indicator row ---
    if show_rsi:
        current_indicator_row += 1
        # Assumed function
        _, rsi_values = rsi_calculation(data)
        fig.add_trace(go.Scatter(x=rsi_values.index, y=rsi_values, name="RSI", line=dict(color="#785cff")),
                      row=current_indicator_row, col=1)
        fig.add_hline(y=70, line_width=1, line_dash="dash", opacity=0.7, line_color="#cc3333", name="Overbought", row=current_indicator_row, col=1)
        fig.add_hline(y=30, line_width=1, line_dash="dash", opacity=0.7, line_color="#33cc33", name="Oversold", row=current_indicator_row, col=1)
        fig.update_yaxes(title_text="RSI", row=current_indicator_row, col=1, range=[0, 100])
        
    # --- Add Daily Returns to the next available indicator row ---
    if show_returns:
        current_indicator_row += 1
        returns = daily_returns(data)
        fig.add_trace(go.Bar(x=returns.index, y=returns, name="Daily Returns (%)", marker=dict(color="#9467bd")),
                      row=current_indicator_row, col=1)
        fig.update_yaxes(title_text="Daily Returns (%)", row=current_indicator_row, col=1)

    # --- Main Price Chart and Overlays (always in the last row) ---
    fig.add_trace(go.Scatter(x=prices.index, y=prices, name="Close Price", mode="lines", line=dict(color="#1f77b4")),
                  row=price_row, col=1, secondary_y=False)

    # Simple Moving Average (SMA)
    if show_sma:
        # Assumed function
        ma = _compute_smas(prices, windows=(sma_window,))[sma_window]
        fig.add_trace(go.Scatter(x=ma.index, y=ma, name=f"SMA {sma_window}", mode="lines", line=dict(color="#ff7f0e")),
                      row=price_row, col=1, secondary_y=False)

    # Consecutive Upward and Downward Runs - Based on Daily Price Changes
    if show_runs:
        # Identify consecutive runs based on close-to-close changes
        runs_df, run_stats = identify_consecutive_runs(data)
        
        # Display run statistics for Trend Runs
        display_run_statistics(run_stats)
        
        # Add run segments to the chart
        if not runs_df.empty:
            for _, run in runs_df.iterrows():
                # Get price data for this run period
                run_start_idx = run['start_idx']
                run_end_idx = run['end_idx']
                run_dates = prices.index[run_start_idx:run_end_idx + 1]
                run_prices = prices.iloc[run_start_idx:run_end_idx + 1]
                
                # Set color and styling based on direction
                if run['direction'] == 'Upward':
                    color = '#22C55E'  # Green for upward runs
                    symbol = '📈'
                    line_width = 4
                else:
                    color = '#EF4444'  # Red for downward runs  
                    symbol = '📉'
                    line_width = 4
                
                # Add colored line segment for this run
                fig.add_trace(go.Scatter(
                    x=run_dates,
                    y=run_prices,
                    mode='lines',
                    line=dict(color=color, width=line_width),
                    name=f"{symbol} {run['direction']} Run ({run['length']} days)",
                    hoverinfo= 'skip',
                    showlegend=False, 
                    opacity=0.8
                ), row=price_row, col=1)
            
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='#22C55E', width=4),
                name=f'📈 Upward Runs ({run_stats["total_upward_runs"]} total)',
                showlegend=True
            ), row=price_row, col=1)
            
            fig.add_trace(go.Scatter(
                x=[None], y=[None], 
                mode='lines',
                line=dict(color='#EF4444', width=4),
                name=f'📉 Downward Runs ({run_stats["total_downward_runs"]} total)',
                showlegend=True
            ), row=price_row, col=1)

    # Max Profit Markers
    if show_maxprofit:
        # Assumed function
        _, filtered_buy_days, filtered_sell_days, filtered_buy_prices, filtered_sell_prices, total_profit = max_profit_calculations(data)
        # Assumed function
        st.success(f"💰 Total Potential Profit: ${total_profit:.2f}")

        fig.add_trace(go.Scatter(
            x=filtered_buy_days, y=filtered_buy_prices, mode='markers', name='Buy',
            marker=dict(symbol='triangle-up', color='green', size=12, line=dict(width=1, color="black"))
        ), row=price_row, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=filtered_sell_days, y=filtered_sell_prices, mode='markers', name='Sell',
            marker=dict(symbol='triangle-down', color='red', size=12, line=dict(width=1, color="black"))
        ), row=price_row, col=1, secondary_y=False)

    # --- Final Layout Updates ---

    # Determine the explicit X-axis name for the main price chart (last row)
    # The X-axis for the Nth row is named 'xaxisN' in Plotly subplots.
    xaxis_key = f'xaxis{price_row}' if price_row > 1 else 'xaxis'

    # Prepare the specific X-axis dictionary with range, selector, and slider
    xaxis_config = dict(
        # Explicitly set the full initial range (helps stabilize the selector)
        range=[prices.index.min(), prices.index.max()],
        
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ],
            # Position the selector right below the title
            y=1.02,  # Above the chart area, below title
            x=0.01,  # Left aligned
            bgcolor="rgba(0,0,0,0)",  # Transparent background
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=12)
        ),
        # Keep rangeslider visible only on this axis
        rangeslider=dict(visible=True)
    )

    # Use a dictionary update to pass the dynamic key to update_layout
    layout_update = {
        xaxis_key: xaxis_config
    }
    
    fig.update_layout(
        hovermode="x unified",
        dragmode="pan",
        showlegend=True,
        title=f"{ticker} — Combined Analysis",
        template='plotly_white',
        height=800,
        width=1200,
        **layout_update  # Apply the dynamically keyed X-axis configuration
    )
    # Update y-axis titles for the main chart
    fig.update_yaxes(title_text="Price", row=price_row, col=1, secondary_y=False)
    
    # Remove rangeslider and rangeselector from individual x-axes to prevent conflicts
    for i in range(1, num_rows + 1):
        is_price_row = (i == price_row)
        
        fig.update_xaxes(
            # The rangeslider is defined in the layout now, but this ensures no others are visible
            rangeslider_visible=False if not is_price_row else True,  
            # Explicitly hide the rangeselector on all indicator rows
            rangeselector_visible=False if not is_price_row else True,
            row=i, col=1
        )

    return fig


# ---------------------------
# Streamlit App Layout
# ---------------------------
st.set_page_config(page_title="Stockie", layout="wide")

st.title("📊 Stockie: Your Stock Analysis Dashboard")
st.write("Stock analysis made easy! Analyse stock performance with advanced metrics and visualizations")

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
    # --- Feature checkboxes (layer onto SAME chart) ---
    c1, c2, c3 = st.columns(3)
    with c1:
        cb_sma   = st.checkbox("SMA", value=False)
        cb_runs  = st.checkbox("Trend Runs", value=False)
    with c2:
        cb_macd  = st.checkbox("MACD", value=False)
        cb_rets  = st.checkbox("Daily Returns", value=False)
    with c3:
        cb_maxp  = st.checkbox("Max Profit", value=False)
        cb_rsi   = st.checkbox("RSI", value=False)
    
    if cb_sma:
        sma_window = st.slider(
            'SMA Window',
            min_value=1,
            max_value=200,
            value=50,
            step=1
        )
    else:
        # If the checkbox is not ticked, set a default value for sma_window
        sma_window = 50
    
    fig = plotly_combined_chart(
        data, ticker,
        sma_window=sma_window,
        show_sma=cb_sma,
        show_runs=cb_runs,
        show_maxprofit=cb_maxp,
        show_returns=cb_rets,
        show_macd=cb_macd,
        show_rsi=cb_rsi
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Explanations shown ONLY for enabled features; readable in dark/light themes ---
    exp = []
    if cb_sma:   exp.append("**SMA**: Average price over N days to smooth noise.")
    if cb_runs:  exp.append("**Trend Runs**: Consecutive upward/downward days based on close-to-close price changes. Shows total count and longest streaks for each direction.")
    if cb_macd:  exp.append("**MACD**: 12/26 EMA spread with 9-EMA signal; histogram above/below 0 shows momentum bias.")
    if cb_rsi:   exp.append("**RSI**: 0–100 scale; >70 overbought, <30 oversold (guide band shown).")
    if cb_rets:  exp.append("**Daily Returns**: % change each day (bottom secondary axis).")
    if cb_maxp:  exp.append("**Max Profit**: Buy low/sell high points (▲ buy, ▼ sell) for top 10% profitable trades. Click on Table below to see all trades.")
    if exp: st.info("**Feature Explanations**\n\n- " + "\n- ".join(exp))
    
    if cb_maxp: 
        trades_df, *_ = max_profit_calculations(data)  # ✅ only keep the first element (DataFrame)
        with st.expander("📊 View Trade Details"):
            st.dataframe(
                trades_df.style.format({
                    "Buy Price": "{:.2f}",
                    "Sell Price": "{:.2f}", 
                    "Profit": "{:.2f}"
                }),
                use_container_width=True
            )

else:
    if ticker_selection != "Others":
        st.info("Select a company ticker first to see analysis options.")
