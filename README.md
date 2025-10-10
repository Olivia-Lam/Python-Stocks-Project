# STOCKIE: stock analysis

An educational web application for analyzing stock market trends using technical indicators. Built with Python, Streamlit, and Plotly for interactive data visualization.

## 🎯 Overview

This project provides a user-friendly interface to explore stock price patterns through four core technical analysis functions:

- **Simple Moving Average (SMA)** - Smooth price data to identify trends
- **Trend Runs** - Detect consecutive upward/downward price movements
- **Daily Returns** - Calculate percentage change day-to-day
- **Max Profit** - Find optimal buy/sell points (theoretical)

**Plus bonus indicators:** MACD and RSI for advanced analysis.

> **Educational Focus:** Clean, readable code with detailed documentation. Perfect for learning algorithmic trading concepts without the "black box" of commercial platforms.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Olivia-Lam/Python-Stocks-Project-Lab-P4-2.git
cd Python-Stocks-Project-Lab-P4-2

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

### Usage

1. Select a stock ticker (AAPL, TSLA, GOOGL, or enter your own)
2. Check boxes to enable indicators (SMA, Trend Runs, Daily Returns, etc.)
3. Adjust the SMA window size with the slider
4. Interact with the chart - zoom, pan, and hover for details

## 📊 Dataset

- **Source:** Yahoo Finance via `yfinance` API
- **Default Period:** 3 years of daily trading data (~756 days)
- **Data Columns:** Open, High, Low, Close, Volume
- **Primary Analysis:** Close prices (adjusted for splits/dividends)

## Core Functions

### Simple Moving Average (SMA)
Calculates the average price over a rolling window to smooth noise and reveal trends.

```python
# Optimized O(n) implementation using cumulative sum
sma = _compute_smas(prices, windows=(50,))
```

### Trend Runs
Identifies consecutive days of price increases (upward runs) or decreases (downward runs).

```python
runs_df, stats = identify_consecutive_runs(data)
# Returns: total runs, longest streaks, average length
```

**Stats:** Total upward/downward runs, longest streaks, average duration

### Daily Returns
Computes percentage change between consecutive closing prices.

```python
daily_return = (current_price - previous_price) / previous_price * 100
```

**Formula:** $(P_t - P_{t-1}) / P_{t-1} \times 100$

### Max Profit
Finds all local minima (buy) and maxima (sell) for theoretical maximum profit.

```python
trades_df, total_profit = max_profit_calculations(ticker)
# Displays top 10% most profitable trades on chart
```

## 📁 Project Structure

```
stock-analysis-toolkit/
├── main.py                     # Streamlit app entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── [functions embedded in main.py]
    ├── download_stock_data()   # Data acquisition
    ├── _compute_smas()         # SMA calculation
    ├── identify_consecutive_runs()  # Trend detection
    ├── daily_returns()         # Volatility calculation
    ├── max_profit_calculations()    # Optimal trading
    ├── macd_calculations()     # MACD indicator
    ├── rsi_calculation()       # RSI indicator
    └── plotly_combined_chart() # Visualization
```

## 👥 Team Contributions

| Team Member | Primary Contribution |
|-------------|---------------------|
| **DANIEL TAY ZHU HAO** | Daily Returns |
| **AVRIL LEONG KE EN** | Upwards and Downwards Run |
| **CHIA TING HUI WILEEN** | Web Interface, Test Cases |
| **EMMANUEL CHOW JIE WEI** | Max Profit, MACD and RSI |
| **OLIVIA LAM XUAN EN** | SMA, Zoom, Combined Charts |
