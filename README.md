# 📈 Stock Market Trend Analysis (Beginner-Friendly)

## 📌 Project Overview
This repository contains a **Python-based stock analysis toolkit** focused on core time-series analytics rather than ML. It helps you download market data, visualize **Simple Moving Average (SMA)**, highlight **Upward/Downward Runs**, and plot **Simple Daily Returns**.

> ✅ Designed for learning: the code is small, readable, and modular.  
> 🧩 Extendable: commented templates are included for **Max Profit**, **MACD**, and **RSI** if you want to expand.

---

## 📊 Dataset Used
- **Source:** Yahoo Finance via the `yfinance` API  
- **Default Lookback:** up to **3 years** (`period='3y'`)  
- **Columns used:** `Open, High, Low, Close, Volume` (we primarily use `Close` here)

---

## ✅ Requirements Mapping

| Requirement | Where it’s implemented |
|---|---|
| Compute **SMA** for a window | `plot_simple_moving_average` *(TODO placeholder — see “Next Steps”)* |
| **Upward/Downward Runs**: visualize runs using SMA(50) trend shading | `plot_upward_downward_runs` |
| **Simple Daily Returns**: \((P_t - P_{t-1})/P_{t-1} \times 100\) | `plot_simple_daily_returns` |
| Visualization: price vs **SMA** on same chart | `plot_simple_moving_average` *(to implement)* |
| Visualization: highlight **runs** on chart | `plot_upward_downward_runs` |

> 💡 The repo includes **commented** templates for:
> - `max_profit_calculations` (LeetCode “Best Time to Buy and Sell Stock II”)
> - `macd_calculations`
> - `rsi_calculation`  

---

## 🤝 Contributions
| Team Member | Assigned Part |
|---|---|
| DANIEL TAY ZHU HAO | Daily Returns |
| AVRIL LEONG KE EN | Upwards and Downwards Run |
| CHIA TING HUI WILEEN | Web Interface, Test Case |
| EMMANUEL CHOW JIE WEI | Max Profit, Macd, RSI |
| OLIVIA LAM XUAN EN | SMA, Zoom, Combined Charts |
