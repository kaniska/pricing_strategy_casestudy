## PunchhMart Project Report: 
https://docs.google.com/document/d/160cJcj77YpMgLgxfxVaIy6yFoeMmXnCmr_Dd-wUpiqc/edit?usp=sharing  

## Exploratory Data Analysis
ItemSalesPriceEDA.ipynb

## Profit Maximization strategy by incresing price using OR Tools Library
cd PriceOptimizerORTools

python PriceOptimizer.py

Initial total price: $15155.00

Optimal total price: $18186.00

Initial total profit: $3285.90

Optimal total profit: $10929.95

-------------------------------

## Profit Maximization strategy using Business Rules for Price Variations based on different buckets of price determinants 
PriceOptimizer.py contains the main Business Logic.

kaniska_mac$ docker-compose up -d

Starting Pricing-Strategy ... done

kaniska_mac$ docker-compose ps
      Name             Command        State           Ports         
--------------------------------------------------------------------

Pricing-Strategy   python mainV2.py   Up      0.0.0.0:8081->8088/tcp

http://0.0.0.0/8081

Loads initial data and displays optimal price and profit in the app

Profit without Optimization: 3285.90

Net Estimated Profit: 4772.95 , LIFT by [45.26 %]

Increase in Total Pricing: 879.47, [5.80 %]

Increase in Total Units Sold: 18.76 %, [5 %]



## Profit Maximization strategy using Demand Function
ProfitEstimatorWithDemandFunction

DemandFunctionProfitEstimator.py

It generates simulated price within required range {base_p*0.9 , base_p*1.2}

It applies OLS and find Intercept & Coefficient for Demand Function

Then Demad Function is applied to a new set of Prices for the item

