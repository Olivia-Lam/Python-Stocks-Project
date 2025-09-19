import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


# Downloading Stock Data Function
def download_stock_data(ticker, period='3y'):
    """
    Download stock data for the given ticker and period
    """
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data


# Simple Moving Average Function
def plot_simple_moving_average(ticker, period = '3y'):
    
    pass


# Upward Downward Runs Function
def plot_upward_downward_runs(ticker, period='3y'):

    data = download_stock_data(ticker, period)
    prices = data['Close']
    
    ma_50 = prices.rolling(window=50).mean()
    trend = prices > ma_50
    
    plt.figure(figsize=(14, 6))
    
    plt.plot(prices.index, prices, 'k-', linewidth=1.5, label='Closing Price')
    plt.plot(prices.index, ma_50, 'b-', linewidth=2, alpha=0.7, label='50-Day MA')
    
    plt.fill_between(prices.index, prices, ma_50, where=trend, 
                    alpha=0.3, color='green', label='Upward Trend')
    plt.fill_between(prices.index, prices, ma_50, where=~trend, 
                    alpha=0.3, color='red', label=
                    'Downward Trend')
    
    plt.title(f'Upward & Downward Runs for {ticker}', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()
    
    
# Simple Daily Returns Function
def plot_simple_daily_returns(ticker, period= '3y'):
    pass

def plot_max_profit_calculations(ticker, period = '3y'):
    pass
    
    
# Main Application
if __name__ == "__main__":
    ticker_symbol = input("Input Stock code: ")
    
    plot_upward_downward_runs(ticker_symbol)
