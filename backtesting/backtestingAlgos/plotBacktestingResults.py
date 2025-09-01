import matplotlib.pyplot as plt
import pandas as pd

def aggregateTotalPortfolioValue(tracking_df_arr):
    value_cols = [df['cumulative_return'] for df in tracking_df_arr]
    returns_df = pd.DataFrame(value_cols).T
    returns_df['portfolio_return'] = returns_df.sum(axis=1)
    returns_df['portfolio_return%'] =returns_df['portfolio_return']*100 

    return returns_df

def plotResults(tracking_df_arr):
    output_df = aggregateTotalPortfolioValue(tracking_df_arr)

    plt.figure(1,figsize=(12,8))
    plt.plot(output_df['portfolio_return%'])

    plt.title('Total Portfolio Return Value Over Time')
    plt.xlabel('Days Since First Trade')
    plt.ylabel('Total Portfolio Return (%)')

    plt.grid(True)
    plt.show()

