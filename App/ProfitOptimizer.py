import pandas as pd
import os
import numpy as np

"""
This code provides the methods to calculate profit 
"""

def calc_profit(df, price_change_val, price_reduction_flag=True):
    original_unit_price = df['unit_price']
    price_change_multiplier = 1 + price_change_val

    # For iterantion#1 increment = 0, price_change_multiplier = 1
    # Note that price adjustment is calculated against the original_unit_price
    adjusted_price = original_unit_price * price_change_multiplier

    print('Original Net price ', np.sum(original_unit_price), 'Adjusted Net price', np.sum(adjusted_price))

    # Avg units sold at base price in one transaction
    original_units_sold = df['units_sold']

    # Calculate additional units for given % of price reduction
    incr_sales = df['incr_sales']
    incr_cvr = df['incr_cvr']
    incr_cvr_default = 0.05
    sales_incr_per_tx = 0

    ##Approach 1
    # Add up incr_cvr and incr_sales as an indicator overall increase in the volume per unit
    # if (price_reduction_flag):
    #      # consider incr_sales only for price reduction
    #     perc_incr_units_sold = incr_sales * abs(price_change_val)*10 + incr_cvr * abs(price_change_val)*10
    # else:
    #     perc_incr_units_sold = incr_cvr_default * abs(price_change_val)*10
    #
    # sales_multipler = 1 + perc_incr_units_sold
    # adjusted_volume = original_units_sold * (sales_multipler)

    ##Approach 2
    # Consider incr_cvr as a boost in more transactions by customers for an item
    # let incr_sales  = increase in avg units sold per transaction
    # and incr_cvr => increase in transactions per item
    # so we need to multiply the changes induced by incr_sales & incr_cvr to compute the overall boost in avg units sold per item
    if (price_reduction_flag):
        sales_incr_per_tx = incr_sales*abs(price_change_val)*10
        customer_incr = incr_cvr*abs(price_change_val)*10
    else:
        customer_incr = incr_cvr_default

    sales_multipler = 1 + sales_incr_per_tx
    transaction_multiplier = 1 + customer_incr
    adjusted_volume = original_units_sold * sales_multipler * transaction_multiplier

    print('Original total units sold ', np.sum(original_units_sold), 'Adjusted total units sold',
      np.sum(adjusted_volume))

    original_cost = df['unit_cost']  # unit cost

    orig_profit = (original_unit_price * original_units_sold) - (original_cost * original_units_sold)
    profit_wo_price_change_effect = np.sum(original_unit_price * original_units_sold) - np.sum( original_cost * original_units_sold)

    # Now that price and volume have been adjusted, its time to compute new profit
    est_profit = (adjusted_price * adjusted_volume) - (original_cost * adjusted_volume)
    profit_with_price_change_effect = np.sum(adjusted_price * adjusted_volume) - np.sum(original_cost * adjusted_volume)

    print('profit_wo_price_change_effect ', profit_wo_price_change_effect, 'profit_with_price_change_effect',
      profit_with_price_change_effect)

    df['price_change_val'] = price_change_val
    df['price_change_multiplier'] = price_change_multiplier
    df['est_price'] = adjusted_price
    df['est_volume'] = adjusted_volume
    df['original_profit'] = orig_profit
    df['est_profit'] = est_profit

    price_comp = df[['price_change_val', 'price_change_multiplier', 'unit_price', 'est_price','est_volume','est_profit']]
    print(price_comp)

    return profit_wo_price_change_effect, profit_with_price_change_effect


def calc_price_reduction(df, initial_reductions, incr_cvr_array, incr_sales_array, initial_profit):
    print('initial profit =', initial_profit)

    final_profit = 0
    intermediate_change = initial_reductions

    reduction_flag = True

    for i in range(0, 250):
        # ToDo consider a combination of hi/lo price and hi/lo unit sld in this pricing strategy
        if intermediate_change[i] >= -0.09 and intermediate_change[i] <= 0.19:
            boost = 0
            if incr_cvr_array[i] >= 0.457650 or incr_sales_array[i] >= 0.096171:
                boost = 0.003 # maximum reduction for products with higher propensity to boost sales and conversion
            elif incr_cvr_array[i] >= 0.366100 or incr_sales_array[i] >= 0.090585:
                boost = 0.002 # moderate reduction for products with higher propensity to boost converstion and moderate increment in sales
            elif incr_cvr_array[i] >= 0.25600 or incr_sales_array[i] >= 0.07658:
                boost = 0.001
            elif incr_cvr_array[i] >= 0.136608 or incr_sales_array[i] >= 0.05098572:
                boost = 0
            elif incr_cvr_array[i] < 0.136608 and incr_sales_array[i] < 0.05098572:
                boost = -0.05 # try aggressive price increase
            else:
                if (df['units_sold'][i] > 1.4 & df['unit_price'][i] > 80):
                    boost = -0.01
                elif (df['units_sold'][i] < 1.4 & df['unit_price'][i] < 30):
                    boost = 0.003
                else:
                    boost = 0.001
            intermediate_change[i] = intermediate_change[i] - boost

        if intermediate_change[i] < -0.10:
            intermediate_change[i] = 0.09
        elif intermediate_change[i] > 0.20:
            intermediate_change[i] = 0.19
    ###

    if (reduction_flag >= 0):
        reduction_flag = False

    profit_wo_price_change_effect, final_profit = calc_profit(df, intermediate_change, reduction_flag)
    if final_profit > initial_profit:
        final_reductions = intermediate_change
    else:
        final_reductions = intermediate_change
        final_profit = initial_profit

    print('final profit = ', final_profit)
    return final_reductions, final_profit


def findOptimalPrice(csv_files_path):
    pd.options.display.float_format = '{:,.2f}'.format
    df = pd.read_csv(csv_files_path)

    price_change_default_val = np.array(df.shape[0] * [0.0])
    current_price_change = price_change_default_val

    converged = False
    max_iter = 0

    print("====================================")
    profit_wo_price_change_effect, current_profit = calc_profit(df, current_price_change, False)

    print("Original profit", profit_wo_price_change_effect)

    incr_cvr_array = df['incr_cvr'].to_numpy()
    incr_sales_array = df['incr_sales'].to_numpy()

    while not converged and max_iter < 10:
        print("====================================")
        next_price_change, next_profit = calc_price_reduction(df, current_price_change, incr_cvr_array,
                                                              incr_sales_array,
                                                              current_profit)
        if (current_profit == next_profit):
            converged = True
        else:
            current_profit = next_profit
            current_price_change = next_price_change

        max_iter += 1

    #
    pd.options.display.float_format = lambda x: '{:.0f}'.format(x) if int(x) == x else '{:,.2f}'.format(x)

    df['est_price'] = df["unit_price"] * (1 + next_price_change)
    final_item_price_data = df[['item_id', 'unit_price', 'est_price', 'units_sold', 'est_volume', 'original_profit' ,'est_profit']]
    df.round(2)
    print('Old Net profit', current_profit)
    print('New Net profit', next_profit)

    change_in_total_price = np.sum(df['est_price']) - np.sum(df['unit_price'])
    change_in_total_units_sold = np.sum(df['est_volume']) - np.sum(df['units_sold'])

    profit_with_price_change_effect = next_profit
    perc_change_in_price = 100 * (change_in_total_price/ np.sum(df['unit_price']))
    perc_lift_in_profit = 100 * ((profit_with_price_change_effect - profit_wo_price_change_effect) / profit_wo_price_change_effect)
    perc_change_in_sales = 100 * (change_in_total_units_sold/ np.sum(df['units_sold']))

    #profit_spike_price_elasticity_df =df.loc[(df["est_price"] < df["unit_price"]) & (df["est_profit"] > df["original_profit"])]
    #print(np.sum(profit_spike_price_elasticity_df["est_profit"] - profit_spike_price_elasticity_df["original_profit"]))

    return final_item_price_data, profit_wo_price_change_effect, profit_with_price_change_effect, change_in_total_price, perc_lift_in_profit, perc_change_in_price, change_in_total_units_sold, perc_change_in_sales

