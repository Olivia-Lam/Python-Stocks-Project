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
    
# Main Application
if __name__ == "__main__":
    ticker_symbol = input("Input Stock code: ")
    
    plot_upward_downward_runs(ticker_symbol)
