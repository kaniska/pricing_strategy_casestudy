import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pandas as pd
import os
import numpy as np
import base64

# -------------------------- functions ---------------------------- #
def build_banner():
    return html.Div(
        id='banner',
        className='banner',
        children=[
            html.Img(src=app.get_asset_url('dash_logo.png')),
        ],
    )


def calc_profit(price_change_val, price_reduction_flag=True):

    original_unit_price = df['unit_price']
    # increment => {-0.1,0.2}
    price_change_multiplier = 1 + price_change_val
    # For iterantion#1 increment = 0, price_change_multiplier = 1
    adjusted_price = original_unit_price * price_change_multiplier

    print('Original Net price ', np.sum(original_unit_price), 'Adjusted Net price', np.sum(adjusted_price))

    original_units_sold = df['units_sold']  # Avg units sold at base price in one transaction
    perc_incr_units_sold = 0  # For iteration#1 perc_incr_units_sold = 0

    # Calculate additional units for given % of price reduction
    incr_sales = df['incr_sales']
    incr_cvr = df['incr_cvr']
    incr_cvr_default = 0.05

    if (price_reduction_flag):
        # consider incr_sales only for price reduction
        perc_incr_units_sold = incr_sales * abs(price_change_val) + incr_cvr * abs(price_change_val)
    else:
        perc_incr_units_sold = incr_cvr_default * abs(price_change_val)

    sales_multipler = 1 + perc_incr_units_sold
    adjusted_volume = original_units_sold * (sales_multipler)

    # print('sales_multipler ',sales_multipler)

    print('Original total units sold ', np.sum(original_units_sold), 'Adjusted total units sold',
          np.sum(adjusted_volume))

    original_cost = df['unit_cost']  # unit cost

    profit_wo_price_change_effect = np.sum(original_unit_price * original_units_sold) - np.sum(
        original_cost * original_units_sold)

    # For every 10% reduction in price, there may be a some increase in conversion leading to increase in profit

    profit_with_price_change_effect = np.sum(adjusted_price * adjusted_volume) - np.sum(original_cost * adjusted_volume)

    print('profit_wo_price_change_effect ', profit_wo_price_change_effect, 'profit_with_price_change_effect',
          profit_with_price_change_effect)

    df['price_change_val']=price_change_val
    df['price_change_multiplier']=price_change_multiplier
    df['adjusted_price']=adjusted_price

    price_comp = df[['price_change_val','price_change_multiplier','unit_price','adjusted_price']]
    print(price_comp)

    return profit_wo_price_change_effect, profit_with_price_change_effect


def calc_price_reduction(initial_reductions, incr_cvr_array, incr_sales_array, initial_profit):
    # print('initial reductions =',initial_reductions)
    print('initial profit =', initial_profit)
    # intermediate_profit = 0
    final_profit = 0
    decrements = initial_reductions

    for i in range(0, 250):
        if decrements[i] >= -0.09 and decrements[i] <= 0.19:
            boost = 0
            if incr_cvr_array[i] >= 0.457650 or incr_sales_array[i] >= 0.096171:
                boost = 0.006
            elif incr_cvr_array[i] >= 0.366100 or incr_sales_array[i] >= 0.090585:
                boost = 0.004
            elif incr_cvr_array[i] >= 0.25600 or incr_sales_array[i] >= 0.07658:
                boost = 0.002
            elif incr_cvr_array[i] >= 0.136608 or incr_sales_array[i] >= 0.05098572:
                boost = 0.001 #maximum reduction for products with higher propensity to boost sales and conversion
            elif incr_cvr_array[i] > 0.136608 and incr_sales_array[i] < 0.05098572:
                boost = 0.0 #moderate reduction for products with higher propensity to boost converstion and moderate increment in sales
            elif incr_cvr_array[i] < 0.136608 and incr_sales_array[i] > 0.05098572:
                boost = -0.1 #moderate reduction for products with higher propensity to boost converstion and moderate increment in sales
            elif incr_cvr_array[i] == 0:
                boost = -0.2
            else:
                boost = -0.15
            decrements[i] = decrements[i] - boost

        # print(i,"{:.3f}".format(decrements[i]))

        if decrements[i] < -0.10:
            decrements[i] = 0.09
        elif decrements[i] > 0.20:
            decrements[i] = 0.19
    ###
    profit_wo_price_change_effect, final_profit = calc_profit(decrements, True)
    if final_profit > initial_profit:
        final_reductions = decrements
    else:
        final_reductions = decrements
        final_profit = initial_profit

    # print('final reductions =',final_reductions)
    print('final profit = ', final_profit)
    return final_reductions, final_profit


# -------------------------- CALC OPTIMAL PRICE ---------------------------- #

def findOptimalPrice():
    price_change_default_val = np.array(df.shape[0] * [0.0])

    converged = False
    current_price_change = price_change_default_val
    # converged = true is a condition when the price_change vector remains the same
    print("====================================")
    profit_wo_price_change_effect, current_profit = calc_profit(current_price_change, False)

    print("Original profit", profit_wo_price_change_effect)

    incr_cvr_array = df['incr_cvr'].to_numpy()
    incr_sales_array = df['incr_sales'].to_numpy()

    while not converged:
        print("====================================")
        next_price_change, next_profit = calc_price_reduction(current_price_change, incr_cvr_array, incr_sales_array,
                                                              current_profit)
        if (current_profit == next_profit):
            converged = True
            break
        else:
            current_profit = next_profit
            current_price_change = next_price_change

    df['optimal_price'] = df["unit_price"]*(1+next_price_change)
    final_result = df[['item_id','unit_price','optimal_price']]

    print('Old Net profit', current_profit)
    print('New Net profit', next_profit)
    return final_result , next_profit
# -------------------------- PROJECT DASHBOARD ---------------------------- #

csv_files_path = os.path.join('data/product_salesV1.csv')
df = pd.read_csv(csv_files_path)
final_price_list, final_profit = findOptimalPrice()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, assets_folder='assets')
server = app.server

app.config.suppress_callback_exceptions = True

dash_text = '''Net profit '''+str(final_profit)



app.layout = html.Div(children=[
    html.H1(
        children=[
            build_banner(),
            html.P(
                id='instructions',
                children=dash_text),
        ]
    ),
    dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in final_price_list.columns],
    data=final_price_list.to_dict('records'),
    fixed_rows={'headers': True},
    #page_size=10,  # we have less data in this example, so setting to 20
    style_table={'height': '300px', 'width': '600px', 'overflowY': 'auto'},
    style_cell={'minWidth': 95, 'width': 95, 'maxWidth': 95}
)

])

# -------------------------- MAIN ---------------------------- #


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8081, debug=False, use_reloader=False)
