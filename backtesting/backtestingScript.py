import argparse
from pathlib import Path
import os
import pandas as pd
import sys
import numpy as np

# Use relative imports for modules within the same package
from . import generatePredictionDataframes as GPD
from .backtestingAlgos.plotBacktestingResults import plotResults
from .backtestingAlgos import loadSyntheticData


# [x] - calculate the final value for each data frame
    # [x] - calculate the return on each trade
        # [x] - use df.shift then calculate close pct dif

    # [x] - register how much of a stock to buy
    # [x] - register how much of a stock to sell
        # [x] - check that we have the correct amount of stock to sell
        # [x] - if not then sell everything we do have then discard the rest of the sell order


# keep a rolling average share price
    # store as tuple? (ticker,num shares,average share price)
    # then update everytime i buy by keeping rolling average
        # weight the number of shares already owned against the number that are going to be bought then average their price


def runBackTesting(df, starting_capital):
    portfolio = {
        'sharesOwned': 0.0,
        'avgSharePrice': 0.0,
        'total_return': 0.0,
        'cash': starting_capital,
        'portfolio_val': starting_capital
    }

    tracking_rows = []
    last_row = None
    for row in df.itertuples():
        if row.Indicator == 1:
            order_val = min(row.OrderVal, portfolio['cash']) # Don't spend cash you don't have
            if order_val > 0:
                N_new_shares = np.round(order_val / row.Close, 5)
                if N_new_shares > 0:
                    # FIX: Use the actual `order_val` spent for the calculation
                    portfolio['avgSharePrice'] = (portfolio['avgSharePrice'] * portfolio['sharesOwned'] + order_val) / (portfolio['sharesOwned'] + N_new_shares)
                    portfolio['sharesOwned'] += N_new_shares
                    portfolio['cash'] -= order_val

        # SELL LOGIC 
        elif row.Indicator == -1:
            if portfolio['sharesOwned'] > 0.0:
                N_sold_shares = min(np.round(row.OrderVal / portfolio['avgSharePrice'], 5), portfolio['sharesOwned'])
                if N_sold_shares > 0:
                    cash_from_sale = N_sold_shares * row.Close
                    return_on_trade = cash_from_sale - (N_sold_shares * portfolio['avgSharePrice'])
                    
                    portfolio['total_return'] += return_on_trade
                    portfolio['cash'] += cash_from_sale
                    portfolio['sharesOwned'] -= N_sold_shares

                    if portfolio['sharesOwned'] <= 1e-9:
                        portfolio['sharesOwned'] = 0.0 # Set to 0 to be clean
                        portfolio['avgSharePrice'] = 0.0
        
        # Update portfolio value for the current day
        portfolio['portfolio_val'] = portfolio['cash'] + (portfolio['sharesOwned'] * row.Close)
        
        metrics_dict = portfolio.copy()
        metrics_dict['Date'] = row.Date
        tracking_rows.append(metrics_dict)
        last_row = row

    # FINAL LIQUIDATION 
    if portfolio['sharesOwned'] > 1e-9 and last_row is not None:
        cash_from_sale = portfolio['sharesOwned'] * last_row.Close
        return_on_trade = cash_from_sale - (portfolio['sharesOwned'] * portfolio['avgSharePrice'])
        
        # Update final portfolio state
        portfolio['total_return'] += return_on_trade
        portfolio['cash'] += cash_from_sale
        portfolio['sharesOwned'] = 0
        portfolio['portfolio_val'] = portfolio['cash'] # Final value is just cash
        
        # Update the last recorded metrics with the final liquidated state
        if tracking_rows:
            tracking_rows[-1].update(portfolio)

    tracking_df = pd.DataFrame(tracking_rows)
    # The final profit/loss
    final_return = portfolio['portfolio_val'] - starting_capital
    return final_return, tracking_df



def getDayOnDayPctDif(df,target_col):
    df[f'prev{target_col}PctDif'] = df[target_col].pct_change() * 100
    return df

#intialise dictionary to store how much of each ticker we currently own
def initialisePortfolioDict(df_arr):
    portfolio = {}
    for df in df_arr:
        ticker = df.head(1)['Ticker'].values[0]
        if ticker not in portfolio:
            portfolio[ticker] = 0

        else:
            print('ERROR > Same Ticker has Multiple Dataframes in the System')

    return portfolio

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Back Testing System the runs on an Already Generated Dataframe')
    parser.add_argument('--data_folder',
                        help='The folder that the data to be back tested is stored in',
                        required=True)
    parser.add_argument('--starting_capital','-sc',
                        type=float,
                        help='The initial capital for the portfolio.',
                        required=True)
    parser.add_argument('--plot_results',
                        help='Plot the Returns of the portfolio on a graph',
                        action='store_true')

    parser.add_argument('--horizon',help='Number of days to Simulate',default=30,type=int)


    args = parser.parse_args()

####### generate the model prediction data frames here

    # for debugging and initial development
    # df_arr = loadSyntheticData.run(args.data_folder)
    
    df_arr = GPD.run(args.data_folder,args.starting_capital,args.horizon)

####### end of generating the model prediction data frames

    print('Calculating Model Profitability')
    total_portfolio_profit = 0
    portfolio_tracking_df_arr = []
    for df in df_arr:
        df = df.copy()
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Allocate a portion of the starting capital to each backtest
        capital_per_stock = args.starting_capital / len(df_arr)
        profit, tracking_df = runBackTesting(df, capital_per_stock)
        total_portfolio_profit += profit
        portfolio_tracking_df_arr.append(tracking_df)

    print(f'Total Strategy Profit: ${total_portfolio_profit:,.2f}')

    if args.plot_results:
        daily_total_portfolio_value = {}
        
        # Sum portfolio values across all individual backtests for each day
        for df in portfolio_tracking_df_arr:
            if not df.empty and 'Date' in df.columns and 'portfolio_val' in df.columns:
                for _, row in df.iterrows():
                    date = row['Date']
                    value = row['portfolio_val']
                    daily_total_portfolio_value[date] = daily_total_portfolio_value.get(date, 0) + value

        if daily_total_portfolio_value:
            # Create a single DataFrame for the total portfolio's value over time
            total_tracking_df = pd.DataFrame(
                list(daily_total_portfolio_value.items()),
                columns=['Date', 'portfolio_val']
            ).sort_values(by='Date').reset_index(drop=True)

            # --- Correct Cumulative Return Calculation ---
            # Calculate return based on the total aggregated portfolio value vs. the total starting capital.
            total_tracking_df['cumulative_return'] = (total_tracking_df['portfolio_val'] / args.starting_capital) - 1

            # Pass the single, aggregated DataFrame to the plotting function
            plotResults([total_tracking_df])
        else:
            print("No tracking data available to plot.")


