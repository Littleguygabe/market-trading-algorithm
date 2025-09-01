


def generateBuyRatio(risk_df):
    CVar_loss_percentile = 5
    VaR_reward_percentile = 100 - CVar_loss_percentile

    reward_row = risk_df[risk_df['VaR_Percentile%']==VaR_reward_percentile]
    risk_row = risk_df[risk_df['VaR_Percentile%']==CVar_loss_percentile]

    reward_pct_change = reward_row['VaR_Val_Change%'].values
    risk_pct_change = risk_row['CVaR_Val_Change%'].values
    
    risk_val_change = risk_row['CVaR_Val_Change$'].values

    reward_risk_ratio = abs(reward_pct_change/risk_pct_change)
    
    return reward_risk_ratio,risk_val_change

def generateSellRatio(risk_df):
    VaR_risk_percentile = 5
    CVaR_reward_percentile = 100 - VaR_risk_percentile

    reward_row = risk_df[risk_df['VaR_Percentile%']==CVaR_reward_percentile]
    risk_row = risk_df[risk_df['VaR_Percentile%']==VaR_risk_percentile]

    reward_pct_change = reward_row['CVaR_Val_Change%'].values
    risk_pct_change = risk_row['VaR_Val_Change%'].values

    risk_val_change = risk_row['VaR_Val_Change$'].values

    reward_risk_ratio = abs(reward_pct_change/risk_pct_change)

    return reward_risk_ratio,risk_val_change

def run(risk_df,mlpt,last_close_price):

    min_ratio_threshold = 2.0

    buy_risk_ratio,buy_risk_val = generateBuyRatio(risk_df)
    sell_risk_ratio,sell_risk_val= generateSellRatio(risk_df)
    # print(buy_risk_ratio,sell_risk_ratio)
    if buy_risk_ratio<min_ratio_threshold and sell_risk_ratio<min_ratio_threshold:
        order_val = 0
        indicator = 0
        nStocks = 0
    
    elif buy_risk_ratio>sell_risk_ratio:
        # buy stock
        nStocks = abs(mlpt/buy_risk_val)
        order_val = nStocks*last_close_price
        indicator = 1
        order_val = order_val[0][0]

    else:
        # sell stock
        nStocks = abs(mlpt/sell_risk_val)
        order_val = nStocks*last_close_price
        indicator = -1
        order_val = order_val[0][0]


    # print(f'ORDER > {nStocks} shares @ ${last_close_price} per share, total order value: ${order_val}, indicator: {indicator}')

    return (order_val,indicator)