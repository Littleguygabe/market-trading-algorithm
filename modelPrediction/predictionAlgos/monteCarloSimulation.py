import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
from tabulate import tabulate

def run(mu_forecast, sigma_forecast, forecast_horizon, n_sims, last_close_price,verbose=0):
    if verbose>-1:
        print('>--------------------')
        print('\033[1mRunning monteCarloSimulation.py\033[0m')
    mu = mu_forecast / 100
    sigma = sigma_forecast / 100

    price_paths = np.zeros((forecast_horizon, n_sims))
    price_paths[0] = last_close_price
    
    for t in range(1, forecast_horizon):
        Z = np.random.standard_normal(n_sims)
        price_paths[t] = price_paths[t-1] * np.exp(
            (mu - 0.5 * sigma**2) + sigma * Z
        )
    return price_paths

def displayPriceAnalytics(data, last_close, verbose):
    last_close = np.squeeze(last_close)
    final_prices = data[-1,:]
    
    sns.histplot(final_prices, bins=100, kde=True, stat="density", label='Distribution of Final Prices')

    kde = sns.kdeplot(final_prices).get_lines()[0].get_data()
    mode_index = np.argmax(kde[1])
    mode_price = kde[0][mode_index]
    
    mean_price = np.mean(final_prices)
    median_price = np.median(final_prices)
    if verbose >1:
        plt.figure(1, figsize=(12, 7))

        plt.axvline(mode_price, color='red', linestyle='--', label=f'Mode (Most Likely): ${mode_price:.2f}')
        plt.axvline(mean_price, color='green', linestyle='--', label=f'Mean (Average): ${mean_price:.2f}')
        plt.axvline(median_price, color='orange', linestyle='--', label=f'Median (50th Percentile): ${median_price:.2f}')
        plt.title('Distribution of Simulated Final Stock Prices')
        plt.xlabel('Final Price ($)')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

    sorted_prices = np.sort(final_prices)
    n_sims = len(sorted_prices)
    
    percentiles = np.arange(1, 100)
    var_price_levels = np.percentile(final_prices, percentiles)

    cvar_price_levels = []
    for p in percentiles:
        if p < 50:
            index = int(n_sims * p / 100)
            cvar_level = sorted_prices[:index].mean()
            cvar_price_levels.append(cvar_level)
        else:
            index = int(n_sims * p / 100)
            cvar_level = sorted_prices[index:].mean()
            cvar_price_levels.append(cvar_level)
    
    cvar_price_levels = np.array(cvar_price_levels)
    
    VaR_values_change = var_price_levels - last_close
    CVaR_values_change = cvar_price_levels - last_close

    if verbose >1:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(VaR_values_change, percentiles, color='blue', label='Value at Risk (VaR)')
        ax.plot(CVaR_values_change, percentiles, color='red', linestyle='--', label='Conditional VaR (CVaR)')
        ax.set_title('Risk Profile: VaR and CVaR', fontsize=16)
        ax.set_xlabel('Potential Value Change ($)', fontsize=12)
        ax.set_ylabel('Percentile', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        plt.tight_layout()

    BOLD = '\033[1m'
    END = '\033[0m'

    if verbose>-1:
        print(f'{BOLD}The last close price was: ${last_close}')
        print(f"The most likely price (Mode) is approximately: ${mode_price:.2f}")
        print(f"The average price (Mean) is: ${mean_price:.2f}")
        print(f"The median price (Median) is: ${median_price:.2f}{END}")

    if verbose >1:
        plt.show()

    risk_df = pd.DataFrame()
    risk_df['VaR_Percentile%'] = percentiles
    risk_df['VaR_Price_Level$'] = var_price_levels
    risk_df['VaR_Val_Change$'] = risk_df['VaR_Price_Level$'] - last_close
    risk_df['VaR_Val_Change%'] = (risk_df['VaR_Val_Change$'] / last_close) * 100
    
    risk_df['CVaR_Price_Level$'] = cvar_price_levels
    risk_df['CVaR_Val_Change$'] = risk_df['CVaR_Price_Level$'] - last_close
    risk_df['CVaR_Val_Change%'] = (risk_df['CVaR_Val_Change$'] / last_close) * 100
    
    risk_df['Occurence_Prob%'] = np.where(risk_df['VaR_Val_Change$'] < 0, risk_df['VaR_Percentile%'], 100 - risk_df['VaR_Percentile%'])
    risk_df['Direction'] = np.where(risk_df['VaR_Val_Change%'] < 0, '-', '+')
    
    risk_df = round(risk_df, 3)

    if verbose >0:
        print(f'Value at Risk Analytic Dataframe:')
        print(tabulate(risk_df, headers='keys', tablefmt='fancy_grid', showindex=False))

    return risk_df