from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from typing import Tuple, Dict, Optional, List

# ---------------------------
# Download Stock Data Function
# ---------------------------
def download_stock_data(ticker: str, period: str = '3y') -> Optional[pd.DataFrame]:
    """
    Download stock data for the given ticker and period with error handling.
    
    Parameters:
        ticker (str): Stock ticker symbol
        period (str): Time period for historical data
        
    Returns:
        pd.DataFrame: Historical stock data, or None if download fails
        
    Raises:
        ValueError: If ticker is empty or invalid
    """
    try:
        # Validate ticker input is not empty
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a non-empty string")
        
        if ticker.strip() == "":
            raise ValueError("Ticker cannot be empty or whitespace")
        
        # Download data using yfinance
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        # Check if data was successfully retrieved
        if data.empty:
            raise ValueError(f"No data available for ticker '{ticker}' with period '{period}'")
        
        # Validate that required columns exist
        required_columns = ['Close']
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return data
        
    except ValueError as ve:
        st.error(f"Validation Error: {ve}")
        return None
    except Exception as e:
        st.error(f"Error downloading data for '{ticker}': {str(e)}")
        return None


# ---------------------------
# Analysis Functions
# ---------------------------
def _compute_smas(prices: pd.Series, windows: tuple = (12, 50, 200)) -> Dict[int, pd.Series]:
    """
    O(n) sliding-window SMAs using one cumulative sum pass with error handling.
    
    Parameters:
        prices (pd.Series): Price data
        windows (tuple): Window sizes for SMA calculation
        
    Returns:
        dict: Dictionary mapping window size to SMA Series
        
    Raises:
        ValueError: If prices is empty or windows are invalid
    """
    try:
        # Validate that prices data exists and is not empty
        if prices is None or len(prices) == 0:
            raise ValueError("Prices data is empty")
        
        if not isinstance(prices, pd.Series):
            raise TypeError("Prices must be a pandas Series")
        
        # Validate windows parameter
        if not windows or not isinstance(windows, (tuple, list)):
            raise ValueError("Windows must be a non-empty tuple or list")
        
        # Check each window size is valid
        for w in windows:
            if not isinstance(w, int) or w <= 0:
                raise ValueError(f"Window size must be a positive integer, got {w}")
            if w > len(prices):
                raise ValueError(f"Window size {w} exceeds data length {len(prices)}")
        
        # Convert to float64 and check for all NaN values
        s = prices.astype("float64")
        if s.isna().all():
            raise ValueError("All price values are NaN")
        
        # Calculate cumulative sum for efficient SMA computation
        csum = s.cumsum()
        out = {}
        
        # Calculate SMA for each window size
        for w in windows:
            try:
                # SMA formula: (cumsum - cumsum_shifted) / window
                sma = (csum - csum.shift(w)) / w
                
                # For the first w-1 values, use expanding mean
                if w > 1:
                    expanding_mean = s.expanding(min_periods=1).mean()
                    sma.iloc[:w-1] = expanding_mean.iloc[:w-1]
                
                out[w] = sma
                
            except Exception as e:
                st.warning(f"Warning: Error computing SMA for window {w}: {e}")
                continue
        
        # Ensure at least one SMA was computed successfully
        if not out:
            raise ValueError("Failed to compute any SMAs")
        
        return out
        
    except (ValueError, TypeError) as e:
        st.error(f"SMA Calculation Error: {e}")
        raise
    except Exception as e:
        st.error(f"Unexpected error in SMA calculation: {e}")
        raise


def simple_moving_average(ticker: str, window: int = 50, period: str = '3y'):
    """
    Generate matplotlib plot of closing price with SMA overlay.
    
    Parameters:
        ticker (str): Stock ticker symbol
        window (int): SMA window size
        period (str): Time period for data
        
    Returns:
        matplotlib.figure.Figure or None: Plot figure if successful
        
    Raises:
        ValueError: If parameters are invalid
    """
    try:
        # Validate window parameter is reasonable
        if not isinstance(window, int) or window <= 0:
            raise ValueError(f"Window must be a positive integer, got {window}")
        
        if window > 500:
            raise ValueError(f"Window size {window} is too large (max 500)")
        
        # Download stock data
        data = download_stock_data(ticker, period)
        if data is None or data.empty:
            st.error(f"Cannot generate SMA plot: No data for '{ticker}'")
            return None
        
        prices = data['Close']
        
        # Ensure we have enough data points for the window size
        if len(prices) < window:
            raise ValueError(f"Insufficient data: {len(prices)} points available, but window size is {window}")
        
        # Compute SMA
        smas = _compute_smas(prices, windows=(window,))
        ma = smas[window]
        
        # Create matplotlib figure with custom styling
        fig, ax = plt.subplots(figsize=(14, 6), facecolor='#e6e6e6')
        ax.set_facecolor('#e6e6e6')
        for s in ax.spines.values():
            s.set_visible(False)
        ax.grid(False)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.tick_params(axis='y', right=True, left=False, colors='#222')
        ax.tick_params(axis='x', colors='#222')

        # Plot closing price and SMA
        ax.plot(prices.index, prices, linewidth=1.2, label='Close')
        ax.plot(ma.index, ma, linewidth=1.2, label=f'SMA {window}')

        # Add labels and title
        ax.legend(frameon=False, loc='upper left')
        ax.set_title(f'{ticker} — Simple Moving Average ({window}-day)')
        ax.set_ylabel('Price')
        
        return fig
        
    except ValueError as ve:
        st.error(f"Error: {ve}")
        return None
    except Exception as e:
        st.error(f"Unexpected error creating SMA plot: {e}")
        return None


def daily_returns(data: pd.DataFrame) -> Optional[pd.Series]:
    """
    Calculate daily percentage returns with error handling.
    
    Parameters:
        data (pd.DataFrame): Stock data with 'Close' column
        
    Returns:
        pd.Series: Daily returns as percentages, or None if calculation fails
    """
    try:
        # Validate input data
        if data is None or data.empty:
            raise ValueError("Input data is empty")
        
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        if len(data) < 2:
            raise ValueError("Need at least 2 data points to calculate returns")
        
        current_price = data['Close']
        
        # Check for and warn about non-positive prices
        if (current_price <= 0).any():
            st.warning("Warning: Data contains non-positive prices")
        
        # Calculate previous day's price
        previous_price = data['Close'].shift(1)
        
        # Calculate daily return percentage
        # Handle potential division by zero with numpy error state
        with np.errstate(divide='ignore', invalid='ignore'):
            daily_return = (current_price - previous_price) / previous_price
            daily_return = daily_return * 100.0
        
        # Replace infinity values with NaN
        daily_return = daily_return.replace([np.inf, -np.inf], np.nan)
        
        return daily_return
        
    except ValueError as ve:
        st.error(f"Daily Returns Error: {ve}")
        return None
    except Exception as e:
        st.error(f"Unexpected error calculating daily returns: {e}")
        return None


def identify_consecutive_runs(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Identify consecutive upward and downward runs with error handling.
    
    Parameters:
        data (pd.DataFrame): Stock data with 'Close' column
        
    Returns:
        tuple: (runs_df, stats_dict) or (empty_df, zero_stats) if error
    """
    try:
        # Validate input data
        if data is None or data.empty:
            raise ValueError("Input data is empty")
        
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        if len(data) < 2:
            raise ValueError("Need at least 2 data points to identify runs")
        
        prices = data['Close']
        
        # Calculate daily price changes
        price_changes = prices.diff()
        
        # Classify each day: 1 for up, -1 for down, 0 for flat
        directions = np.where(price_changes > 0, 1, np.where(price_changes < 0, -1, 0))
        
        # Initialize run tracking variables
        runs = []
        current_direction = None
        run_start = None
        run_length = 0
        
        # Iterate through price changes to identify runs
        for i, direction in enumerate(directions):
            # Skip flat days (no price change)
            if direction == 0:
                continue
            
            # Check if direction changed
            if current_direction != direction:
                # Save the previous run if it exists
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
                
                # Start tracking a new run
                current_direction = direction
                run_start = i
                run_length = 1
            else:
                # Continue current run
                run_length += 1
        
        # Add the final run
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
        
        # Convert runs list to DataFrame
        runs_df = pd.DataFrame(runs)
        
        # Calculate summary statistics
        if not runs_df.empty:
            upward_runs = runs_df[runs_df['direction'] == 'Upward']
            downward_runs = runs_df[runs_df['direction'] == 'Downward']
            
            stats = {
                'total_upward_runs': len(upward_runs),
                'total_downward_runs': len(downward_runs),
                'longest_upward_run': int(upward_runs['length'].max()) if len(upward_runs) > 0 else 0,
                'longest_downward_run': int(downward_runs['length'].max()) if len(downward_runs) > 0 else 0,
                'avg_upward_length': float(upward_runs['length'].mean()) if len(upward_runs) > 0 else 0,
                'avg_downward_length': float(downward_runs['length'].mean()) if len(downward_runs) > 0 else 0
            }
        else:
            # Return zero statistics if no runs found
            stats = {
                'total_upward_runs': 0,
                'total_downward_runs': 0,
                'longest_upward_run': 0,
                'longest_downward_run': 0,
                'avg_upward_length': 0,
                'avg_downward_length': 0
            }
        
        return runs_df, stats
        
    except ValueError as ve:
        st.error(f"Runs Identification Error: {ve}")
        # Return empty DataFrame and zero statistics
        return pd.DataFrame(), {
            'total_upward_runs': 0, 'total_downward_runs': 0,
            'longest_upward_run': 0, 'longest_downward_run': 0,
            'avg_upward_length': 0, 'avg_downward_length': 0
        }
    except Exception as e:
        st.error(f"Unexpected error identifying runs: {e}")
        return pd.DataFrame(), {
            'total_upward_runs': 0, 'total_downward_runs': 0,
            'longest_upward_run': 0, 'longest_downward_run': 0,
            'avg_upward_length': 0, 'avg_downward_length': 0
        }


def display_run_statistics(stats: Dict):
    """
    Display upward and downward run statistics in Streamlit.
    
    Parameters:
        stats (dict): Dictionary containing run statistics
    """
    try:
        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)
        
        # Display upward run statistics
        with col1:
            st.metric(
                "Upward Runs", 
                f"{stats['total_upward_runs']}", 
                f"Longest: {int(stats['longest_upward_run'])} days"
            )
            if stats['avg_upward_length'] > 0:
                st.caption(f"Average length: {stats['avg_upward_length']:.1f} days")
        
        # Display downward run statistics
        with col2:
            st.metric(
                "Downward Runs", 
                f"{stats['total_downward_runs']}", 
                f"Longest: {int(stats['longest_downward_run'])} days"
            )
            if stats['avg_downward_length'] > 0:
                st.caption(f"Average length: {stats['avg_downward_length']:.1f} days")
                
    except Exception as e:
        st.error(f"Error displaying statistics: {e}")


def max_profit_calculations(data: pd.DataFrame) -> Tuple[pd.DataFrame, List, List, List, List, float]:
    """
    Calculate maximum profit using greedy algorithm with error handling.
    Finds optimal buy/sell points assuming perfect foresight.
    
    Parameters:
        data (pd.DataFrame): Stock data with 'Close' column
    
    Returns:
        tuple: (trades_df, filtered_buy_days, filtered_sell_days, 
                filtered_buy_prices, filtered_sell_prices, total_profit)
    """
    try:
        # Validate input data
        if data is None or data.empty:
            raise ValueError("Input data is empty")
        
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        if len(data) < 2:
            raise ValueError("Need at least 2 data points for profit calculation")
        
        prices = data['Close']
        
        # Warn if data contains non-positive prices
        if (prices <= 0).any():
            st.warning("Warning: Data contains non-positive prices")
        
        # Initialize tracking lists
        buy_days = []
        sell_days = []
        prices_bought = []
        prices_sold = []
        trades = []
        total_profit = 0
        i = 0
        n = len(prices)
        price_list = prices.tolist()
        dates = prices.index.tolist()
        
        # Greedy algorithm to find buy/sell points
        while i < n - 1:
            # Find local minimum (buy point)
            # Skip days where price is decreasing or flat
            while i < n - 1 and price_list[i+1] <= price_list[i]:
                i += 1
            if i == n - 1:
                break
            
            # Record buy point
            buy_price = price_list[i]
            buy_day = dates[i]
            buy_days.append(buy_day)
            prices_bought.append(buy_price)
            
            # Find local maximum (sell point)
            # Move forward while price is increasing
            while i < n - 1 and price_list[i+1] >= price_list[i]:
                i += 1
            
            # Record sell point
            sell_price = price_list[i]
            sell_day = dates[i]
            sell_days.append(sell_day)
            prices_sold.append(sell_price)
            
            # Calculate profit from this trade
            profit = sell_price - buy_price
            total_profit += profit

            # Store trade details
            trades.append({
                "Buy Date": buy_day.strftime("%Y-%m-%d"),
                "Buy Price": round(buy_price, 2),
                "Sell Date": sell_day.strftime("%Y-%m-%d"),
                "Sell Price": round(sell_price, 2),
                "Profit": round(profit, 2)
            })

        # Handle case where no profitable trades were found
        if not trades:
            st.warning("Warning: No profitable trades found (prices only decreased)")
            return pd.DataFrame(columns=["Buy Date", "Buy Price", "Sell Date", "Sell Price", "Profit"]), [], [], [], [], 0.0
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades)

        # Filter for top 10% most profitable trades for visualization
        top_trades_threshold = trades_df["Profit"].quantile(0.9)
        filtered = trades_df[trades_df["Profit"] >= top_trades_threshold]
        
        # Extract filtered trade data for plotting
        filtered_buy_days = pd.to_datetime(filtered["Buy Date"]).tolist()
        filtered_sell_days = pd.to_datetime(filtered["Sell Date"]).tolist()
        filtered_buy_prices = filtered["Buy Price"].tolist()
        filtered_sell_prices = filtered["Sell Price"].tolist()
        
        return trades_df, filtered_buy_days, filtered_sell_days, filtered_buy_prices, filtered_sell_prices, round(total_profit, 2)
        
    except ValueError as ve:
        st.error(f"Max Profit Calculation Error: {ve}")
        return pd.DataFrame(), [], [], [], [], 0.0
    except Exception as e:
        st.error(f"Unexpected error in max profit calculation: {e}")
        return pd.DataFrame(), [], [], [], [], 0.0


def macd_calculations(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence) indicator with error handling.
    
    MACD Line = 12-day EMA - 26-day EMA
    Signal Line = 9-day EMA of MACD Line
    Histogram = MACD Line - Signal Line
    
    Parameters:
        data (pd.DataFrame): Stock data with 'Close' column
    
    Returns:
        tuple: (prices, macd_line, signal_line, histogram)
    """
    try:
        # Validate input data
        if data is None or data.empty:
            raise ValueError("Input data is empty")
        
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        # MACD requires at least 26 periods for EMA26 calculation
        if len(data) < 26:
            raise ValueError(f"MACD requires at least 26 data points, got {len(data)}")
        
        prices = data['Close']

        # Calculate 12-day and 26-day Exponential Moving Averages
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()

        # Calculate MACD line (difference between EMAs)
        macd_line = ema_12 - ema_26
        
        # Calculate signal line (9-day EMA of MACD line)
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        # Calculate histogram (difference between MACD and signal)
        histogram = macd_line - signal_line

        return prices, macd_line, signal_line, histogram
        
    except ValueError as ve:
        st.error(f"MACD Calculation Error: {ve}")
        # Return empty series with proper dtype
        empty = pd.Series(dtype=float)
        return empty, empty, empty, empty
    except Exception as e:
        st.error(f"Unexpected error calculating MACD: {e}")
        empty = pd.Series(dtype=float)
        return empty, empty, empty, empty


def rsi_calculation(data: pd.DataFrame, window: int = 14) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate RSI (Relative Strength Index) indicator with error handling.
    
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over the window period
    
    Parameters:
        data (pd.DataFrame): Stock data with 'Close' column
        window (int): RSI calculation window (default 14 days)
        
    Returns:
        tuple: (prices, rsi_values)
    """
    try:
        # Validate input data
        if data is None or data.empty:
            raise ValueError("Input data is empty")
        
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        # Validate window parameter
        if not isinstance(window, int) or window <= 0:
            raise ValueError(f"Window must be a positive integer, got {window}")
        
        # RSI requires at least window+1 data points
        if len(data) < window + 1:
            raise ValueError(f"RSI requires at least {window + 1} data points, got {len(data)}")
        
        prices = data['Close']

        # Calculate daily price changes
        difference = prices.diff()

        # Separate gains (positive changes) and losses (negative changes)
        gain = difference.where(difference > 0, 0)
        loss = -difference.where(difference < 0, 0)

        # Calculate average gain and average loss over the window
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()

        # Calculate Relative Strength (RS)
        # Handle division by zero case when avg_loss is 0
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / avg_loss
            # Replace infinity with large number (when avg_loss = 0)
            rs = rs.replace([np.inf], 100)
            
            # Calculate RSI using the formula
            rsi = 100 - (100 / (1 + rs))
        
        # Ensure RSI values are within valid 0-100 range
        rsi = rsi.clip(0, 100)

        return prices, rsi
        
    except ValueError as ve:
        st.error(f"RSI Calculation Error: {ve}")
        empty = pd.Series(dtype=float)
        return empty, empty
    except Exception as e:
        st.error(f"Unexpected error calculating RSI: {e}")
        empty = pd.Series(dtype=float)
        return empty, empty


def plotly_sma_zoom(
    data, ticker, window=50,
    show_runs=False, show_returns=False, show_maxprofit=False,
    show_momentum=False
):
    """
    Create interactive Plotly chart with SMA and optional overlays.
    
    Parameters:
        data (pd.DataFrame): Stock data
        ticker (str): Stock ticker symbol
        window (int): SMA window size
        show_runs (bool): Show trend runs overlay
        show_returns (bool): Show daily returns
        show_maxprofit (bool): Show buy/sell markers
        show_momentum (bool): Show MACD indicator
    
    Returns:
        plotly.graph_objects.Figure: Interactive chart
    """
    try:
        prices = data['Close']
        volume = data['Volume'] if 'Volume' in data.columns else None
        
        # Calculate SMA
        ma = _compute_smas(prices, windows=(window,))[window]

        # Calculate MACD for momentum indicator
        _, macd_line, signal_line, histogram = macd_calculations(data)
        
        # Determine volume bar colors based on price direction
        up = prices.diff().fillna(0) >= 0
        vol_colors = np.where(up, "#26a69a", "#ef5350") if volume is not None else None

        # Set row heights based on whether momentum is shown
        row_heights = [0.32, 0.68] if show_momentum else [0.0001, 0.9999]

        # Create subplot figure with 2 rows
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
            row_heights=row_heights,
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
        )

        # Add MACD indicator to top panel if enabled
        if show_momentum:
            # Add histogram bars
            fig.add_trace(
                go.Bar(x=histogram.index, y=histogram, name="MACD Hist",
                       marker=dict(color=np.where(histogram >= 0, "rgba(55, 170, 70, 0.6)", "rgba(230, 80, 80, 0.6)")),
                       opacity=0.8),
                row=1, col=1
            )
            # Add MACD and signal lines
            fig.add_trace(go.Scatter(x=macd_line.index, y=macd_line, name="MACD", mode="lines"), row=1, col=1)
            fig.add_trace(go.Scatter(x=signal_line.index, y=signal_line, name="Signal", mode="lines"), row=1, col=1)
            # Add zero reference line
            fig.add_hline(y=0, line_width=1, line_dash="dot", opacity=0.4, row=1, col=1)

        # Add price and SMA lines to main panel
        fig.add_trace(go.Scatter(x=prices.index, y=prices, name="Close", mode="lines"), row=2, col=1)
        fig.add_trace(go.Scatter(x=ma.index, y=ma, name=f"SMA {window}", mode="lines"), row=2, col=1)

        # Add trend runs overlay if enabled
        if show_runs:
            ma50 = _compute_smas(prices, windows=(50,))[50]
            trend_up = (prices > ma50)
            
            # Add baseline for upward trend fill
            fig.add_trace(go.Scatter(
                x=prices.index, 
                y=ma50, 
                mode="lines",
                line=dict(width=0), 
                showlegend=False, 
                hoverinfo="skip",
                name="MA50_baseline"
            ), row=2, col=1)
            
            # Add green fill for upward trend
            fig.add_trace(go.Scatter(
                x=prices.index, 
                y=prices.where(trend_up), 
                name="Upward Trend",
                mode="lines", 
                line=dict(width=0),
                fill="tonexty", 
                fillcolor="rgba(34, 139, 34, 0.3)",
                hovertemplate="Upward Trend<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"
            ), row=2, col=1)
            
            # Add baseline for downward trend fill
            fig.add_trace(go.Scatter(
                x=prices.index, 
                y=ma50, 
                mode="lines",
                line=dict(width=0), 
                showlegend=False, 
                hoverinfo="skip",
                name="MA50_baseline2"
            ), row=2, col=1)
            
            # Add red fill for downward trend
            fig.add_trace(go.Scatter(
                x=prices.index, 
                y=prices.where(~trend_up), 
                name="Downward Trend",
                mode="lines", 
                line=dict(width=0),
                fill="tonexty", 
                fillcolor="rgba(220, 20, 60, 0.3)",
                hovertemplate="Downward Trend<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"
            ), row=2, col=1)
            
            # Add MA50 reference line
            fig.add_trace(go.Scatter(
                x=ma50.index, 
                y=ma50, 
                name="MA50 Reference",
                mode="lines", 
                line=dict(color="#4169E1", width=1.5, dash="dot"),
                hovertemplate="MA50: $%{y:.2f}<extra></extra>"
            ), row=2, col=1)

        # Add volume bars on secondary y-axis
        if volume is not None:
            fig.add_trace(
                go.Bar(x=volume.index, y=volume, name="Volume",
                       marker=dict(color=vol_colors), opacity=0.7),
                row=2, col=1, secondary_y=True
            )
            fig.update_yaxes(title_text="Volume", row=2, col=1, secondary_y=True, showgrid=False)

        # Add range selector buttons and slider
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
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            hovermode="x unified", 
            dragmode="pan", 
            showlegend=True,
            margin=dict(l=10, r=10, t=40, b=10),
            title=f"{ticker} — Zoomable Price / SMA ({window}-day)"
        )
        
        # Update axis labels
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

    # Use .ewm to calculate average gain & loss
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()

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
    """
    Create comprehensive multi-panel Plotly chart with various technical indicators.
    
    Parameters:
        data (pd.DataFrame): Stock data
        ticker (str): Stock ticker symbol
        sma_window (int): SMA window size
        show_sma (bool): Show SMA overlay
        show_runs (bool): Show consecutive run segments
        show_maxprofit (bool): Show buy/sell markers
        show_returns (bool): Show daily returns panel
        show_macd (bool): Show MACD panel
        show_rsi (bool): Show RSI panel
    
    Returns:
        plotly.graph_objects.Figure: Multi-panel interactive chart
    """
    try:
        prices = data["Close"]

        # Calculate number of rows needed based on selected indicators
        num_rows = 1  # Main price chart
        if show_macd:
            num_rows += 1
        if show_rsi:
            num_rows += 1
        if show_returns:
            num_rows += 1
        
        # Calculate row heights (indicators get 20%, price gets remainder)
        indicator_heights = 0.20
        price_height = 1 - (num_rows - 1) * indicator_heights
        row_heights = [indicator_heights] * (num_rows - 1) + [price_height]
        
        # Set up subplot specifications
        specs = [[{"secondary_y": False}]] * num_rows

        # Initialize subplot figure
        fig = make_subplots(
            rows=num_rows, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            row_heights=row_heights,
            specs=specs
        )

        # Price chart is always in the last row
        price_row = num_rows
        current_indicator_row = 0

        # Add MACD indicator panel if enabled
        if show_macd:
            current_indicator_row += 1
            _, macd_line, signal_line, histogram = macd_calculations(data)
            
            # Add MACD histogram
            fig.add_trace(go.Bar(
                x=histogram.index, y=histogram, name="MACD Hist",
                marker=dict(color=["#26a69a" if h > 0 else "#ef5350" for h in histogram]),
                opacity=0.5
            ), row=current_indicator_row, col=1)
            
            # Add MACD and signal lines
            fig.add_trace(go.Scatter(x=macd_line.index, y=macd_line, name="MACD Line", line=dict(color="#2962FF")),
                          row=current_indicator_row, col=1)
            fig.add_trace(go.Scatter(x=signal_line.index, y=signal_line, name="Signal Line", line=dict(color="#FF6D00")),
                          row=current_indicator_row, col=1)
            
            # Update axis and add zero line
            fig.update_yaxes(title_text="MACD", row=current_indicator_row, col=1)
            fig.add_hline(y=0, line_width=1, line_dash="dash", opacity=0.4, line_color="#888", row=current_indicator_row, col=1)

        # Add RSI indicator panel if enabled
        if show_rsi:
            current_indicator_row += 1
            _, rsi_values = rsi_calculation(data)
            
            # Add RSI line
            fig.add_trace(go.Scatter(x=rsi_values.index, y=rsi_values, name="RSI", line=dict(color="#785cff")),
                          row=current_indicator_row, col=1)
            
            # Add overbought and oversold reference lines
            fig.add_hline(y=70, line_width=1, line_dash="dash", opacity=0.7, line_color="#cc3333", name="Overbought", row=current_indicator_row, col=1)
            fig.add_hline(y=30, line_width=1, line_dash="dash", opacity=0.7, line_color="#33cc33", name="Oversold", row=current_indicator_row, col=1)
            
            # Update axis with fixed range
            fig.update_yaxes(title_text="RSI", row=current_indicator_row, col=1, range=[0, 100])
            
        # Add daily returns panel if enabled
        if show_returns:
            current_indicator_row += 1
            returns = daily_returns(data)
            
            # Add returns bar chart
            fig.add_trace(go.Bar(x=returns.index, y=returns, name="Daily Returns (%)", marker=dict(color="#9467bd")),
                          row=current_indicator_row, col=1)
            fig.update_yaxes(title_text="Daily Returns (%)", row=current_indicator_row, col=1)

        # Add main price line to price chart
        fig.add_trace(go.Scatter(x=prices.index, y=prices, name="Close Price", mode="lines", line=dict(color="#1f77b4")),
                      row=price_row, col=1, secondary_y=False)

        # Add SMA overlay if enabled
        if show_sma:
            ma = _compute_smas(prices, windows=(sma_window,))[sma_window]
            fig.add_trace(go.Scatter(x=ma.index, y=ma, name=f"SMA {sma_window}", mode="lines", line=dict(color="#ff7f0e")),
                          row=price_row, col=1, secondary_y=False)

        # Add consecutive runs overlay if enabled
        if show_runs:
            # Identify runs
            runs_df, run_stats = identify_consecutive_runs(data)
            
            # Display run statistics
            display_run_statistics(run_stats)
            
            # Add colored line segments for each run
            if not runs_df.empty:
                for _, run in runs_df.iterrows():
                    # Extract price data for this run
                    run_start_idx = run['start_idx']
                    run_end_idx = run['end_idx']
                    run_dates = prices.index[run_start_idx:run_end_idx + 1]
                    run_prices = prices.iloc[run_start_idx:run_end_idx + 1]
                    
                    # Set color based on run direction
                    if run['direction'] == 'Upward':
                        color = '#22C55E'  # Green
                        symbol = 'Upward'
                        line_width = 4
                    else:
                        color = '#EF4444'  # Red
                        symbol = 'Downward'
                        line_width = 4
                    
                    # Add run segment to chart
                    fig.add_trace(go.Scatter(
                        x=run_dates,
                        y=run_prices,
                        mode='lines',
                        line=dict(color=color, width=line_width),
                        name=f"{symbol} Run ({run['length']} days)",
                        hoverinfo='skip',
                        showlegend=False, 
                        opacity=0.8
                    ), row=price_row, col=1)
                
                # Add legend entries for run types
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='lines',
                    line=dict(color='#22C55E', width=4),
                    name=f'Upward Runs ({run_stats["total_upward_runs"]} total)',
                    showlegend=True
                ), row=price_row, col=1)
                
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], 
                    mode='lines',
                    line=dict(color='#EF4444', width=4),
                    name=f'Downward Runs ({run_stats["total_downward_runs"]} total)',
                    showlegend=True
                ), row=price_row, col=1)

        # Add max profit buy/sell markers if enabled
        if show_maxprofit:
            _, filtered_buy_days, filtered_sell_days, filtered_buy_prices, filtered_sell_prices, total_profit = max_profit_calculations(data)
            
            # Display total profit
            st.success(f"Total Potential Profit: ${total_profit:.2f}")

            # Add buy markers (green triangles pointing up)
            fig.add_trace(go.Scatter(
                x=filtered_buy_days, y=filtered_buy_prices, mode='markers', name='Buy',
                marker=dict(symbol='triangle-up', color='green', size=12, line=dict(width=1, color="black"))
            ), row=price_row, col=1, secondary_y=False)

            # Add sell markers (red triangles pointing down)
            fig.add_trace(go.Scatter(
                x=filtered_sell_days, y=filtered_sell_prices, mode='markers', name='Sell',
                marker=dict(symbol='triangle-down', color='red', size=12, line=dict(width=1, color="black"))
            ), row=price_row, col=1, secondary_y=False)

        # Configure x-axis controls
        xaxis_key = f'xaxis{price_row}' if price_row > 1 else 'xaxis'

        # Set up range selector and slider
        xaxis_config = dict(
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
                y=1.02,
                x=0.01,
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                font=dict(size=12)
            ),
            rangeslider=dict(visible=True)
        )

        # Apply x-axis configuration
        layout_update = {
            xaxis_key: xaxis_config
        }
        
        # Update layout
        fig.update_layout(
            hovermode="x unified",
            dragmode="pan",
            showlegend=True,
            title=f"{ticker} — Combined Analysis",
            template='plotly_white',
            height=800,
            width=1200,
            **layout_update
        )
        
        # Update y-axis for price chart
        fig.update_yaxes(title_text="Price", row=price_row, col=1, secondary_y=False)
        
        # Configure rangeslider visibility for each row
        for i in range(1, num_rows + 1):
            is_price_row = (i == price_row)
            fig.update_xaxes(
                rangeslider_visible=False if not is_price_row else True,  
                rangeselector_visible=False if not is_price_row else True,
                row=i, col=1
            )

        return fig
        
    except Exception as e:
        st.error(f"Error creating combined chart: {e}")
        return None


# ---------------------------
# Streamlit App Layout
# ---------------------------
st.set_page_config(page_title="Stockie", layout="wide")

st.title("STOKIE: Your Stock Analysis Dashboard")
st.write("Stock analysis made easy! Analyse stock performance with advanced metrics and visualizations")

# Step 1: Select stock ticker
ticker_choices = ["-- Select --", "AAPL", "TSLA", "GOOGL", "AMZN", "META", "Others"]
ticker_selection = st.selectbox(
    "Select a Company Ticker",
    ticker_choices,
    index=None,  
    placeholder="Choose a ticker..."
)

# Handle custom ticker input
ticker_input = ticker_selection
if ticker_selection == "Others":
    ticker_input = st.text_input("Type any ticker you want")

# Clean and uppercase ticker
ticker = ticker_input.strip().upper() if ticker_input else None

# Step 2: Validate ticker and download data
data = None
if ticker:
    try:
        data = download_stock_data(ticker)
        if data is None or data.empty:
            st.error(f"No data found for ticker '{ticker}'. Please check and try again.")
            ticker = None
        else:
            st.success(f"Valid ticker '{ticker}' found!")
    except Exception as e:
        st.error(f"Error retrieving data for '{ticker}': {e}")
        ticker = None

# Step 3: Show analysis options if ticker is valid
if ticker and data is not None and not data.empty:
    # Create three columns for checkboxes
    c1, c2, c3 = st.columns(3)
    
    # Column 1: Basic indicators
    with c1:
        cb_sma = st.checkbox("SMA", value=False)
        cb_runs = st.checkbox("Trend Runs", value=False)
    
    # Column 2: Advanced indicators
    with c2:
        cb_macd = st.checkbox("MACD", value=False)
        cb_rets = st.checkbox("Daily Returns", value=False)
    
    # Column 3: Trading analysis
    with c3:
        cb_maxp = st.checkbox("Max Profit", value=False)
        cb_rsi = st.checkbox("RSI", value=False)
    
    # Show SMA window slider only if SMA is enabled
    if cb_sma:
        sma_window = st.slider(
            'SMA Window',
            min_value=1,
            max_value=200,
            value=50,
            step=1
        )
    else:
        sma_window = 50  # Default value
    
    # Generate combined chart with selected features
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
    
    # Display chart if successfully created
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    # Show explanations for enabled features
    exp = []
    if cb_sma:
        exp.append("**SMA**: Average price over N days to smooth noise.")
    if cb_runs:
        exp.append("**Trend Runs**: Consecutive upward/downward days based on close-to-close price changes. Shows total count and longest streaks for each direction.")
    if cb_macd:
        exp.append("**MACD**: 12/26 EMA spread with 9-EMA signal; histogram above/below 0 shows momentum bias.")
    if cb_rsi:
        exp.append("**RSI**: 0-100 scale; >70 overbought, <30 oversold (guide band shown).")
    if cb_rets:
        exp.append("**Daily Returns**: % change each day (bottom secondary axis).")
    if cb_maxp:
        exp.append("**Max Profit**: Buy low/sell high points (triangle up = buy, triangle down = sell) for top 10% profitable trades. Click on Table below to see all trades.")
    
    # Display feature explanations if any are enabled
    if exp:
        st.info("**Feature Explanations**\n\n- " + "\n- ".join(exp))
    
    # Show detailed trade table for max profit analysis
    if cb_maxp:
        trades_df, *_ = max_profit_calculations(data)
        with st.expander("View Trade Details"):
            st.dataframe(
                trades_df.style.format({
                    "Buy Price": "{:.2f}",
                    "Sell Price": "{:.2f}", 
                    "Profit": "{:.2f}"
                }),
                use_container_width=True
            )
else:
    # Show instruction message if no ticker selected
    if ticker_selection != "Others":
        st.info("Select a company ticker first to see analysis options.")