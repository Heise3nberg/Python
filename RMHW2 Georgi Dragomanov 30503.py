# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 20:25:36 2020

@author: Georgi
"""
import pandas_datareader

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from pandas_datareader import data
from math import sqrt

boeing = data.DataReader('BA', 'yahoo',start='1/1/2000')

plt.plot(boeing["Adj Close"])

#CAGR

DAYS = (boeing.index[-1] - boeing.index[1]).days

CAGR = ((((boeing["Adj Close"][-1])/boeing["Adj Close"][1]))**(365.0/DAYS)) - 1

print(CAGR)
MU = CAGR

#Calculate returns:

boeing["Returns"] = boeing["Adj Close"].pct_change()

#Volatility
AVOL = boeing["Returns"].std() * sqrt(252)

print(AVOL)
SIG = AVOL

#Define variables
T = 252
S = boeing["Adj Close"][-1]

price_list = [S]

result = []

daily_returns = np.random.normal((MU/T),(SIG/sqrt(T)),T)+1

#Run 10000 MC simulations

for i in range(10000):
    daily_returns = np.random.normal((MU/T),SIG/sqrt(T),T)+1
    price_list = [S]
    for x in daily_returns:
        price_list.append(price_list[-1]*x)
    plt.plot(price_list)
    result.append(price_list[-1])

#Plot dist of price simulations
plt.figure(dpi=60, figsize=(13, 6))      
plt.hist(result,bins=100)

print("projected value=",round(np.mean(result),2))
expected_return = (np.mean(result)/S) - 1
print("expected return=",expected_return)
print("5% quantile =",np.percentile(result,5))
print("95% quantile =",np.percentile(result,95))

plt.figure(dpi=60, figsize=(13, 6))  
plt.hist(result,bins=100)
plt.axvline(np.percentile(result,5), color='r', linestyle='dashed', linewidth=2)
plt.axvline(np.percentile(result,95), color='r', linestyle='dashed', linewidth=2)







