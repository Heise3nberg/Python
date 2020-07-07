# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 00:22:37 2020

@author: Georgi
"""
from pandas import Series
from matplotlib import pyplot
from numpy import log
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

from pandas import Series
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot as plt
import numpy as np

virus = Series.from_csv("owid-covid-data.csv",header = 0, index_col=0)
 
plt.plot(virus)

#Stationary test (not that its needed...)

from statsmodels.tsa.stattools import adfuller

y = virus.values

result = adfuller(y)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#Transforming to log values
from numpy import log

logdata = log(y)

plt.plot(logdata)

#ACF and PACF Analysis

acf = acf(logdata)
plt.figure()
plot_acf(acf)

pacf = pacf(logdata)
plt.figure()
plot_pacf(pacf)

#Adfuller for logged data

result1 = adfuller(logdata)

print('ADF Statistic: %f' % result1[0])
print('p-value: %f' % result1[1])
print('Critical Values:')
for key, value in result1[4].items():
	print('\t%s: %.3f' % (key, value))


#ARIMA MODEL PREPARATION
model = ARIMA(virus.values,order = (1,2,0))
model_fit = model.fit(disp = 0)
print(model_fit.summary())

residuals = DataFrame(model_fit.resid)
residuals.plot()
residuals.plot(kind='kde')
residuals.describe()

#After sedcond differencing the model shows heteroskedasticity - it's not an ARIMA MODEL!!!!


#TRY GARCH....


#CALCULATE VARIANCE AND VARIANCE ANALISYS
from numpy import sqrt
squared_residuals = np.array((residuals)**2)

plt.figure()
plot_acf(squared_residuals)

plt.figure()
plot_pacf(squared_residuals)

# define GARCH model
model1 = arch_model(train, mean='Zero', vol='GARCH', p=5, q=1)
model_fit1 = model.fit()
yhat = model_fit.forecast(horizon=n_test)



#TO BE CONTINUED
    









