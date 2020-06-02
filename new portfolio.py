# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:25:50 2020

@author: Georgi
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import sqrt

#list of stocks in portfolio
stocks = ['AAPL','AMZN','FB','MSFT','TSLA','NFLX','GOOGL','WMT','BA','KO']

#download daily price data for each of the stocks in the portfolio
data = web.DataReader(stocks,data_source='yahoo',start='01/01/2013')['Adj Close']
data.sort_index(inplace=True)

#convert daily stock prices into daily returns
returns = data.pct_change()

#calculate mean daily return and covariance of daily returns
mean_daily_returns = returns.mean()
daily_volatility = returns.std()
#portfolio weights
weights = 1/10
#calculate portfolio return
portfolio_return = np.sum(mean_daily_returns * weights) * 252
#volatility
portfolio_std_dev = np.sum(returns.std() * weights) * sqrt(252)
#time period
time = 252
#price list
prices = np.array((data[-1:]))

s = np.sum(prices)*weights

#create random walk of returns
daily_returns = np.random.normal((portfolio_return/time),(portfolio_std_dev/sqrt(time)),time)+1

plt.plot(daily_returns)

plt.hist(daily_returns)

#list if price series
price_list = [s]

for x in daily_returns:
    price_list.append(price_list[-1]*x)

plt.plot(price_list)

#
result = []

for i in range(1000):
    daily_returns = np.random.normal((portfolio_return/time),portfolio_std_dev/sqrt(time),time)+1
    price_list = [s]
    for x in daily_returns:
        price_list.append(price_list[-1]*x)
    result.append(price_list[-1])
    plt.plot(price_list)

plt.hist(result,bins=20)

np.mean(result)

percent_increase = (np.mean(result) - s)/s
print(percent_increase)


#########################################################

#set number of runs of random portfolio weights
num_portfolios = 25000

#set up array to hold results
#We have increased the size of the array to hold the weight values for each stock
results = np.zeros((4+len(stocks)-1,num_portfolios))

for i in range(num_portfolios):
    #select random weights for portfolio holdings
    weights = np.array(np.random.random(10))
    #rebalance weights to sum to 1
    weights /= np.sum(weights)
    #calculate portfolio return
    portfolio_return = np.sum(mean_daily_returns * weights) * 252
    #volatility
    portfolio_std_dev = np.sum(returns.std() * weights) * sqrt(252)    
    #store results in results array
    results[0,i] = portfolio_return
    results[1,i] = portfolio_std_dev
    #store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
    results[2,i] = results[0,i] / results[1,i]
    #iterate through the weight vector and add data to results array
    for j in range(len(weights)):
        results[j+3,i] = weights[j]

#convert results array to Pandas DataFrame
results_frame = pd.DataFrame(results.T,columns=['ret','stdev','sharpe',stocks[0],stocks[1],stocks[2],stocks[3],
                                                stocks[4],stocks[5],stocks[6],stocks[7],stocks[8],stocks[9]])


#locate position of portfolio with highest Sharpe Ratio
max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
print(max_sharpe_port)



#########################################################################
newstocks = ['AMZN','AAPL','GOOGL','WMT','TSLA']

data1 = web.DataReader(newstocks,data_source='yahoo',start='01/01/2013')['Adj Close']
data1.sort_index(inplace=True)


newreturns = data1.pct_change()

#calculate mean daily return and covariance of daily returns
mean_daily_returns1 = newreturns.mean()

#portfolio weights
weights1 = 1/6
#calculate portfolio return
portfolio_return1 = np.sum(mean_daily_returns1 * weights1) * 252
#volatility
portfolio_std_dev1 = np.sum(newreturns.std() * weights1) * sqrt(252)
#time period
time = 252
#price list
prices1 = np.array((data1[-1:]))

S = np.sum(prices1)*weights1

#create random walk of returns
daily_returns1 = np.random.normal((portfolio_return1/time),(portfolio_std_dev1/sqrt(time)),time)+1

plt.plot(daily_returns1)

plt.hist(daily_returns1)

#list if price series
price_list1 = [S]

for x in daily_returns1:
    price_list1.append(price_list1[-1]*x)

plt.plot(price_list1)

#
result1 = []

for i in range(1000):
    daily_returns1 = np.random.normal((portfolio_return1/time),portfolio_std_dev1/sqrt(time),time)+1
    price_list1 = [S]
    for x in daily_returns1:
        price_list1.append(price_list1[-1]*x)
    result1.append(price_list1[-1])
    plt.plot(price_list1)

plt.hist(result1,bins=20)

np.mean(result1)

percent_increase1 = (np.mean(result1) - S)/S
print(percent_increase1)





























