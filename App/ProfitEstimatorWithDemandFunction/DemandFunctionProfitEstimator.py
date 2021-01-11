import pandas as pd
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols

pd.options.display.float_format = '{:,.2f}'.format
df = pd.read_csv('data/product_salesV1.csv')
#print(df.head(10))

itemA_base_price = df['unit_price'][0]
itemA_base_quantity = df['units_sold'][0]
itemA_base_cost = df['unit_cost'][0]
itemA_base_profit = itemA_base_quantity*(itemA_base_price-itemA_base_cost)
incr_sales = df['incr_sales'][0]
incr_cvr = df['incr_cvr'][0]
itemA_price_lb = itemA_base_price*0.90
itemA_price_ub = itemA_base_price*1.2
itemA_df = pd.DataFrame.empty
#  Generate 100 random price for the item => random.uniform(itemA_price_lb, itemA_price_ub)
N=100
price_sim = [random.uniform(itemA_price_lb, itemA_price_ub) for i in range(0,N)]
price_sim[0] = itemA_base_price

sales_vol_sim = [0 for i in range(0,N)]
sales_vol_sim[0] = itemA_base_quantity

profit_sim = [0 for i in range(0,N)]
profit_sim[0]=itemA_base_profit

fixed_cost = [itemA_base_cost for i in range(0,N)]
##
# if price_change < 0, then incr_sales = incr_sales*price_change*10 , incr_cvr = incr_cvr*price_change*10 ;
# else incr_cvr = 0.05

for i in range(0, len(price_sim)-1):
    price_change = (price_sim[i + 1] - price_sim[0]) / price_sim[0]

    if(price_change < 0):
        sales_vol_sim[i+1] = sales_vol_sim[0]*(1+incr_sales*abs(price_change)*10)*(1+incr_cvr*abs(price_change)*10)
    else:
        sales_vol_sim[i+1] = sales_vol_sim[0]*1.05

    profit_sim[i+1] = sales_vol_sim[i+1]*(price_sim[i+1] - fixed_cost[0])

d = {'fixed_cost': fixed_cost, 'price_sim': price_sim,'profit_sim': profit_sim,'sales_vol_sim': sales_vol_sim}
itemA_df = pd.DataFrame(data=d)
print(itemA_df[['price_sim','sales_vol_sim', 'profit_sim']].head(50))

# plot demand curve
sns.set(font_scale=1.5,style="white")
sns.lmplot(x="price_sim",y="sales_vol_sim",data=itemA_df, size = 4)
plt.show()

# fit OLS model
model = ols("sales_vol_sim ~ price_sim", data = itemA_df).fit()
# print model summary
print(model.summary())

model_summary_df = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]
C=model_summary_df['coef'].values[0]
S=model_summary_df['coef'].values[1]



# plugging regression coefficients: C= Intercept, S = Price Coeff
#demand_pred = C - S * price

## Run on Test Data
M = 50
fixed_cost2 = [itemA_base_cost for i in range(0,M)]
price_sim2 = [random.uniform(itemA_price_lb, itemA_price_ub) for i in range(0,M)]
d = {'price_sim': price_sim2,'fixed_cost': fixed_cost2,}
itemA_df2 = pd.DataFrame(data=d)
itemA_df2['demand_pred'] = C - S * itemA_df.price_sim
# the profit function becomes
itemA_df2['profit_pred'] = itemA_df2.demand_pred * (itemA_df2.price_sim - itemA_df2.fixed_cost)

# plot
sns.lmplot(x="price_sim",y="demand_pred",data=itemA_df2, size = 4)
plt.show()
sns.lmplot(x="profit_pred",y="demand_pred",data=itemA_df2, size = 4)
plt.show()

# price at which revenue is maximum
max_val = itemA_df2[itemA_df2['profit_pred'] == itemA_df2['profit_pred'].max()]
print(" Max Profit Data: ", max_val)
