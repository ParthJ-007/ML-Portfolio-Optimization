# -*- coding: utf-8 -*-
"""Portfolio Optimization
# **Install and Import Libraries**
"""
"""Project"""

pip install PyPortfolioOpt

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from pypfopt import EfficientFrontier, risk_models, expected_returns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


"""# **Portfolio Statistics Functions**
**Sharpe Ratio** measures the excess return per unit of total volatility, assessing if returns adequately compensate for all risk taken.

**Sortino Ratio** measures the excess return per unit of downside volatility, focusing only on the risk of negative returns below a target.

**Information Ratio** measures the active return per unit of tracking error, quantifying a manager's skill in outperforming a benchmark consistently.

"""

def sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def sortino_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return np.sqrt(252) * excess_returns.mean() / downside_std

def information_ratio(returns, benchmark_returns):
    return_difference = returns - benchmark_returns
    if return_difference.std() == 0 or np.isnan(return_difference.std()):
        return 0.0 # Avoid division by zero
    return np.sqrt(252) * return_difference.mean() / return_difference.std()

"""# **Machine Learning Stock Return Forecast**
We write a function to download the stock data and uses it to train and select the best of two machine learning models, Random Forest and XGBoost, for predicting the next day's stock return.

It creates time-series features like lagged and rolling returns, trains the models on historical data, and chooses the one with the lowest prediction error (MSE).

Finally, it uses the best model, retrained on the full dataset, to provide a realistic, clipped, and annualized forecast of the next day's return.
"""

def generate_best_ml_view(ticker, start_date, end_date, price_col='Close'):
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)[price_col]
    if stock_data.empty or len(stock_data) < 60: return 0.0, "No Model"
    returns = stock_data.pct_change()
    target = returns.shift(-1)
    features = pd.DataFrame(index=returns.index)
    features['lag_1'], features['lag_5'] = returns.shift(1), returns.shift(5)
    features['rolling_mean_10'] = returns.rolling(window=10).mean()
    features['rolling_std_10'] = returns.rolling(window=10).std()
    full_dataset = pd.concat([features, target], axis=1).dropna()
    full_dataset.columns = list(features.columns) + ['target']
    if len(full_dataset) < 30: return 0.0, "No Model"
    X, y = full_dataset.drop('target', axis=1), full_dataset['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=1000, random_state=42, n_jobs=-1, early_stopping_rounds=10)
    }
    best_model_name, lowest_mse = None, float('inf')
    for name, model in models.items():
        if name == 'XGBoost':
            X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, shuffle=False)
            model.fit(X_train_part, y_train_part, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, predictions)
        if mse < lowest_mse:
            lowest_mse, best_model_name = mse, name
    if best_model_name == 'XGBoost': final_model = xgb.XGBRegressor(n_estimators=1000, random_state=42, n_jobs=-1)
    else: final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    final_model.fit(scaler.transform(X), y)
    predicted_daily_return = final_model.predict(scaler.transform(X.iloc[[-1]]))[0]
    annual_return = (1 + predicted_daily_return)**252 - 1
    clipped_return = np.clip(annual_return, -0.9, 3.0)
    return clipped_return, best_model_name

"""# **Black-Litterman Model Functions**

The Black-Litterman model is a portfolio optimization technique that uses Bayesian statistics to combine two inputs: a neutral prior of market equilibrium returns (derived via reverse-optimization) and an investor's subjective views on asset performance.

It aims to produce a final set of expected returns for the Markowitz Mean-Variance Optimization that are more intuitive, stable, and less sensitive to input errors, resulting in well-diversified portfolios that reflect the manager's unique insights.
"""

def calculate_implied_returns(cov_matrix, market_weights, risk_aversion=2.5):
    return risk_aversion * cov_matrix.dot(market_weights)
def black_litterman_adjustment(implied_returns, cov_matrix, investor_views, tau=0.025):
    P, Q, Omega = investor_views['P'], investor_views['Q'], investor_views['Omega']
    tau_cov_inv = np.linalg.inv(tau * cov_matrix)
    P_t_omega_inv_P = P.T @ np.linalg.inv(Omega) @ P
    tau_cov_inv_pi = tau_cov_inv @ implied_returns
    P_t_omega_inv_Q = P.T @ np.linalg.inv(Omega) @ Q
    return np.linalg.inv(tau_cov_inv + P_t_omega_inv_P) @ (tau_cov_inv_pi + P_t_omega_inv_Q)

"""# **Implementing ML Strategies**"""

if __name__ == "__main__":
    # 1. SETUP PARAMETERS
    portfolio = {
        'RELIANCE.NS': 10000.0, # Reliance Industries Limited
        'TCS.NS': 10000.0, # 	Tata Consultancy Services Limited
        'HDFCBANK.NS': 10000.0, # HDFC Bank Limited
        'ICICIBANK.NS': 10000.0, # ICICI Bank Limited
        'BHARTIARTL.NS': 10000.0, # Bharti Airtel Limited
        'SBIN.NS': 10000.0, # State Bank of India
        'INFY.NS': 10000.0, # Infosys Limited
        'LT.NS': 10000.0, # 	Larsen & Toubro Limited
        'ITC.NS': 10000.0, # ITC Limited
        'HINDUNILVR.NS': 10000.0 # Hindustan Unilever Limited
    }
    tickers, market_index = list(portfolio.keys()), '^NSEI'
    training_start, training_end, backtest_start, backtest_end = '2015-05-27', '2020-05-27', '2020-05-27', '2025-09-29'
    risk_free_rate = 0.04
    price_col = 'Close'

    # 2. CALCULATE ALL EXPECTED RETURNS FOR MODELS
    print("--- Calculating Expected Returns for Each Model ---")
    training_data = yf.download(tickers, start=training_start, end=training_end, progress=False)[price_col]
    S = risk_models.sample_cov(training_data)
    mu_hist = expected_returns.mean_historical_return(training_data)
    market_caps = {t: yf.Ticker(t).info.get('marketCap', 0) for t in tickers}
    market_weights = np.array([market_caps.get(t, 0) / sum(market_caps.values()) for t in tickers])
    implied_returns = calculate_implied_returns(S, market_weights)
    P = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, -1, 0, 0, 0, 0, 0, 0]])
    Q = np.array([0.20, 0.03])
    Omega = np.diag(np.diag(P @ S @ P.T))
    bl_adjusted_returns = black_litterman_adjustment(implied_returns, S, {'P': P, 'Q': Q, 'Omega': Omega})
    ml_expected_returns = {}
    for ticker in tickers:
        forecast, model_name = generate_best_ml_view(ticker, training_start, training_end, price_col=price_col)
        ml_expected_returns[ticker] = forecast
    ml_returns_series = pd.Series(ml_expected_returns)

    # 3. DISPLAY EXPECTED RETURNS TABLE WITH CORRECTED PERCENTAGE FORMATTING
    returns_df = pd.DataFrame({
        'Historical (for MVO)': mu_hist.map('{:.2%}'.format),
        'Black-Litterman': pd.Series(bl_adjusted_returns, index=tickers).map('{:.2%}'.format),
        'Machine Learning': ml_returns_series.map('{:.2%}'.format)
    })
    print("\n--- Expected Annual Returns used by Each Technique ---")
    print(returns_df)

    # 4. CALCULATE PORTFOLIO WEIGHTS
    unoptimized_weights = pd.Series({t: 1/len(tickers) for t in tickers})
    ef_mvo = EfficientFrontier(mu_hist, S); ef_mvo.max_sharpe(); weights_mv = pd.Series(ef_mvo.clean_weights())
    ef_bl = EfficientFrontier(bl_adjusted_returns, S); ef_bl.max_sharpe(); weights_bl = pd.Series(ef_bl.clean_weights())
    ef_ml = EfficientFrontier(ml_returns_series, S); ef_ml.max_sharpe(); weights_ml = pd.Series(ef_ml.clean_weights())

    # DISPLAY WEIGHTS TABLE WITH CORRECTED PERCENTAGE FORMATTING
    weights_df = pd.DataFrame({
        'Unoptimized': unoptimized_weights.map('{:.2%}'.format),
        'MVO Max-Sharpe': weights_mv.reindex(unoptimized_weights.index).map('{:.2%}'.format),
        'Black-Litterman': weights_bl.reindex(unoptimized_weights.index).map('{:.2%}'.format),
        'ML Max-Sharpe': weights_ml.reindex(unoptimized_weights.index).map('{:.2%}'.format)
    }).fillna('0.00%')
    print("\n--- Final Portfolio Allocation Percentages ---")
    print(weights_df)

    # 5. BACKTESTING
    print("\n--- Backtesting All Portfolios ---")
    backtest_data_stocks = yf.download(tickers, start=backtest_start, end=backtest_end, progress=False)[price_col]
    backtest_data_market = yf.download(market_index, start=backtest_start, end=backtest_end, progress=False)[price_col].squeeze()
    daily_returns_stocks = backtest_data_stocks.pct_change().dropna()
    daily_returns_market = backtest_data_market.pct_change().dropna()
    returns_unoptimized = daily_returns_stocks.dot(unoptimized_weights)
    returns_market = daily_returns_market
    returns_mv = daily_returns_stocks.dot(weights_mv)
    returns_bl = daily_returns_stocks.dot(weights_bl)
    returns_ml = daily_returns_stocks.dot(weights_ml)

    # 6. CALCULATE & DISPLAY FINAL METRICS
    strategies = {
        "ML Max-Sharpe Portfolio": returns_ml, "Black-Litterman Portfolio": returns_bl,
        "MVO Max-Sharpe Portfolio": returns_mv, "Market Index (Nifty 50)": returns_market,
        "Original Unoptimized Portfolio": returns_unoptimized
    }
    metrics = {}
    print("\n--- Final Performance Metrics ---")
    for name, returns in strategies.items():
        info = information_ratio(returns, daily_returns_market)
        if name == "Market Index (Nifty 50)": info = 1.0
        cum_return = ((1 + returns).cumprod() - 1).iloc[-1]
        metrics[name] = {"Sharpe Ratio": sharpe_ratio(returns, risk_free_rate), "Sortino Ratio": sortino_ratio(returns, risk_free_rate), "Info Ratio": info, "Return": cum_return}
        print(f"\n{name}:\n  Final Cumulative Return: {cum_return:.2%}\n  Sharpe Ratio: {metrics[name]['Sharpe Ratio']:.2f}")

    # 7. PLOTTING
    print("\n--- Generating Final Plot ---")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(18, 10))
    colors = ['cyan', 'magenta', 'white', 'limegreen', 'red']
    for i, (name, returns) in enumerate(strategies.items()):
        linestyle = '--' if name.startswith('Market') else '-'
        ax.plot(((1 + returns).cumprod() - 1) * 100, label=name, color=colors[i], linestyle=linestyle, linewidth=2)
    ax.set_title('Comparative Cumulative Returns of All Strategies', fontsize=18)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Percentage Gain (%)', fontsize=12)
    ax.legend(loc='upper center')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    y_pos = 0.97
    for i, (name, data) in enumerate(metrics.items()):
        text = (f"{name:<30}\n"
                f"{'Sharpe Ratio:':<18}{data['Sharpe Ratio']:.2f}\n"
                f"{'Sortino Ratio:':<18}{data['Sortino Ratio']:.2f}\n"
                f"{'Info Ratio:':<18}{data['Info Ratio']:.2f}\n"
                f"{'Return:':<18}{data['Return']*100:.2f}%")
        ax.text(0.015, y_pos, text, transform=ax.transAxes, fontsize=9, verticalalignment='top',
                fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.4", edgecolor=colors[i], facecolor='black', alpha=0.8))
        y_pos -= 0.13
    plt.tight_layout()
    plt.show()

