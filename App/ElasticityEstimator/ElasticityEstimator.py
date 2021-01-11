import pandas as pd
import os
import numpy as np
from pymc3 import *
import matplotlib.pyplot as plt
import arviz as az

pd.options.display.float_format = '{:,.2f}'.format
df = pd.read_csv('data/product_salesV1.csv')
#print(df.head(10))

df['unit_price_l'] = np.log(df['unit_price'])
df['units_sold_l'] = np.log(df['units_sold'])
df_l = df[df['units_sold_l'] != 0]
df_l = df[df['unit_price_l'] != 0]

df_l.rename(columns={"unit_price_l": "x"}, inplace=True)

with Model() as model:
    GLM.from_formula('units_sold_l ~ x', df_l[['units_sold_l', 'x']], family='normal')
    trace = sample(4000, cores=1)

var_min = df_l['x'].min()
var_max = df_l['x'].max()

plt.plot(df_l['x'], df_l['units_sold_l'], 'x')
plot_posterior_predictive_glm(trace, eval=np.linspace(var_min, var_max, 100))
plt.show()

#traceplot(trace)
#plt.show()

df_sum = summary(trace)