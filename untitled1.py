# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:59:45 2020

@author: Georgi
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import sqrt

stocks = ['AAPL','AMZN','FB','MSFT','TSLA','NFLX','GOOGL','WMT','BA','KO']

data = web.DataReader(stocks,data_source='yahoo',start='01/01/2013')['Adj Close']
data.sort_index(inplace=True)

returns = data.pct_change()

#calculate mean daily return and covariance of daily returns
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()

#number random portfolio weights
num_portfolios = 50000

#set up array to hold results
results = np.zeros((4+len(stocks)-1,num_portfolios))

for i in range(num_portfolios):
    weights = np.array(np.random.random(10))
    weights /= np.sum(weights)
    
    portfolio_return = np.sum(mean_daily_returns * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252)
    
    results[0,i] = portfolio_return
    results[1,i] = portfolio_std_dev
    results[2,i] = results[0,i] / results[1,i]

    for j in range(len(weights)):
        results[j+3,i] = weights[j]


results_frame = pd.DataFrame(results.T,columns=['ret','stdev','sharpe',stocks[0],stocks[1],stocks[2],stocks[3],
                                                stocks[4],stocks[5],stocks[6],stocks[7],stocks[8],stocks[9]])

#locate position of portfolio with highest Sharpe Ratio
max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]


#plot Sharpe Ratio
plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='RdYlBu')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.colorbar()
plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker="X",color='r',s=500)

rec_port = pd.DataFrame(max_sharpe_port[3:])

rec_port.columns = (["weight"])

rec_port = rec_port.drop(rec_port[rec_port.weight < 0.05].index)

y = (1-np.sum(rec_port))+1

rec_port = rec_port["weight"]*float(y)

print(rec_port)






















































