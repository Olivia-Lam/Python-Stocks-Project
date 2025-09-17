
import pandas as pd

from datetime import datetime

import yfinance as yf

## Download 

df_tesla=yf.download('aapl')
aapl= yf.Ticker("aapl")
aapl_historical = aapl.history(start="2023-06-02", end="2025-06-07", interval="1m")
print(aapl_historical)