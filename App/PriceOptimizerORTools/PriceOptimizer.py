from __future__ import print_function
from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np

def main():
    #Load Data
    data = pd.read_csv('data/product_salesV1.csv')
    data_arr = np.array(data)
    # Define basic Linear programming solver
    solver = pywraplp.Solver('SolveStigler', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    # Declare an array to hold price data
    target_price = [[]] * len(data)

    #convert requiret features into np array
    data['initial_profit'] = data['units_sold']*(data['unit_price'] - data['unit_cost'])
    data['adj_quantity'] = data['units_sold']*(1+data['incr_cvr']*10)*(1+data['incr_sales']*10)
    data['adj_profit'] = data['adj_quantity']*((data['unit_price'] - data['unit_cost']))

    incr_cvr = np.array(data['incr_cvr'],dtype=np.float32)
    price_change_multipler = np.array(data.shape[0] * [0.9])
    initial_price =np.array(data['unit_price'],dtype=np.float32)
    adj_quantity = np.array(data['adj_quantity'],dtype=np.float32)

    initial_profit_vals = np.array(data['initial_profit'],dtype=np.float32)
    adj_profit = np.array(data['adj_profit'],dtype=np.float32)

    # Objective: Maximize the Profit by increasing price
    objective = solver.Objective()
    price_lb = [float(i) for i in initial_price*0.9]
    price_ub = [float(i) for i in initial_price*1.2]
    net_adj_profit = np.sum(adj_profit)

    for i in range(0, len(data)):
        target_price[i] = solver.NumVar(price_lb[i], price_ub[i], str(data_arr[i][0]))
        objective.SetCoefficient(target_price[i], float(price_change_multipler[i]))

    objective.SetMaximization()
    #objective.SetMinimization()

    # Create the constraints
    constraints = []
    # New profit >= Initial profit
    constraints.append(solver.Constraint(float(np.sum(initial_profit_vals)),solver.infinity()))
    for i in range(0, len(data)):
        constraints[0].SetCoefficient(target_price[i], float(adj_quantity[i]))

    # Solve!
    status = solver.Solve()

    if status == solver.OPTIMAL:
        optimal_price = 0
        optimal_profit = 0
        for i in range(0, len(data)):
          optimal_price += target_price[i].solution_value()
          if target_price[i].solution_value() > 0:
              # new price for each item
              print('%s, %f, %f' % (data['item_id'][i], initial_price[i], target_price[i].solution_value()))

        for i in range(0, len(data)):
            price_change_frac = (target_price[i].solution_value() - initial_price[i])/initial_price[i]
            if(price_change_frac > 0):
                optimal_quantity = data['units_sold'][i]*(1+data['incr_cvr'][i]*abs(price_change_frac)*10)*(1+data['incr_sales'][i]*abs(price_change_frac)*10)
            else:
                optimal_quantity = data['units_sold'][i]*1.05
            optimal_profit += optimal_quantity * (target_price[i].solution_value() - data['unit_cost'][i])

        print('Initial total price: $%.2f' % (np.sum(initial_price)))
        print('Optimal total price: $%.2f' % (optimal_price))
        print('Initial total profit: $%.2f' % (np.sum(initial_profit_vals)))
        print('Optimal total profit: $%.2f' % (optimal_profit))
    else:  # No optimal solution was found.
        if status == solver.FEASIBLE:
          print('A potentially suboptimal solution was found.')
        else:
          print('The solver could not solve the problem.')

if __name__ == '__main__':
    main()