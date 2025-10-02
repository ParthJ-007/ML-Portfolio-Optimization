# ML-Portfolio-Optimization

## Project Overview

This project implements a sophisticated framework for **quantitative portfolio optimization**, comparing traditional **Mean-Variance Optimization (MVO)** against two advanced methodologies: the **Black-Litterman Model** and an **MVO enhanced with Machine Learning (ML) forecasts**.

It integrates financial theory with cutting-edge data science techniques to generate optimized portfolio weights and rigorously backtests their performance against a benchmark and an unoptimized portfolio. The core objective is to demonstrate how structured investor views (Black-Litterman) and predictive models (ML-MVO) can lead to superior risk-adjusted returns compared to purely historical or equal-weighted approaches.

## Features

- **Data Acquisition:** Uses the `yfinance` library to fetch historical price data for a basket of **Nifty 50 stocks** and the **^NSEI** market index.
- **Multiple Optimization Strategies:**
    - **Mean-Variance Optimization (MVO):** Classic portfolio optimization using historical returns and covariance.
    - **Black-Litterman Model:** Adjusts market equilibrium returns based on structured investor views.
    - **Machine Learning-Enhanced MVO (ML-MVO):** Uses a pre-trained **Random Forest** or **XGBoost** model to forecast the next day's annual return for each asset, which is then used as the expected return input for MVO.
- **Robust Performance Metrics:** Calculates key risk-adjusted metrics like the **Sharpe Ratio**, **Sortino Ratio**, and **Information Ratio** during the backtesting phase.
- **Comparative Backtesting:** Simulates and plots the cumulative returns of all optimized portfolios, the unoptimized portfolio, and the market index over a defined backtesting period.

## Libraries

| Library | Purpose |
| :--- | :--- |
| **yfinance** | Downloads historical stock and index data. |
| **numpy** | Fundamental library for numerical operations. |
| **pandas** | Essential for data structuring and time-series handling. |
| **scikit-learn** | Used for data preparation, scaling, and **Random Forest** regression. |
| **xgboost** | High-performance implementation of Gradient Boosting for return forecasting. |
| **pypfopt** | State-of-the-art library for portfolio optimization (Efficient Frontier). |
| **matplotlib** | Used for generating the final comparative returns plot. |

## Sample Output

The output below reflects the results obtained from training the model from 2015-05-27 to 2020-05-27 and backtesting from 2020-05-27 to 2025-09-29.
---------- **Expected Annual Returns used by Each Technique** ---------
                    Historical (for MVO)  Black-Litterman   Machine Learning
BHARTIARTL.NS                9.27%           6.84%          -90.00%

HDFCBANK.NS                 11.84%           6.27%          300.00%

HINDUNILVR.NS               20.65%           5.28%          -39.51%

ICICIBANK.NS                 1.65%          11.22%          253.81%

INFY.NS                      9.75%           8.34%          -90.00%

ITC.NS                      -0.44%          10.66%          300.00%

LT.NS                       -3.86%           8.10%          300.00%

RELIANCE.NS                 28.18%          10.62%          -75.11%

SBIN.NS                    -11.27%           6.41%          132.31%

TCS.NS                      10.50%           6.73%          -90.00%



-------- **Final Portfolio Allocation Percentages** ---------
              Unoptimized    MVO Max-Sharpe   Black-Litterman   ML Max-Sharpe
RELIANCE.NS        10.00%         44.05%           5.29%         0.00%
TCS.NS             10.00%          6.43%           6.19%         0.00%
HDFCBANK.NS        10.00%          0.00%          10.96%        60.88%
ICICIBANK.NS       10.00%          0.00%           9.68%         0.00%
BHARTIARTL.NS      10.00%          0.00%          20.13%         0.00%
SBIN.NS            10.00%          0.00%           5.27%         0.00%
INFY.NS            10.00%          0.00%          11.88%         0.00%
LT.NS              10.00%          0.00%           6.25%        24.75%
ITC.NS             10.00%          0.00%           8.36%        14.37%
HINDUNILVR.NS      10.00%         49.51%          16.00%         0.00%

--- **Final Performance Metrics** ---

ML Max-Sharpe Portfolio:
  Final Cumulative Return: 191.33%
  Sharpe Ratio: 0.98

Black-Litterman Portfolio:
  Final Cumulative Return: 192.48%
  Sharpe Ratio: 1.25

MVO Max-Sharpe Portfolio:
  Final Cumulative Return: 77.51%
  Sharpe Ratio: 0.50

Market Index (Nifty 50):
  Final Cumulative Return: 164.68%
  Sharpe Ratio: 1.07

Original Unoptimized Portfolio:
  Final Cumulative Return: 205.82%
  Sharpe Ratio: 1.28

<img width="1789" height="989" alt="download" src="https://github.com/user-attachments/assets/a33b4433-0b39-4c48-a5ec-e5773089ec5e" />

