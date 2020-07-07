# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 01:23:56 2020

@author: Georgi
"""

from pandas import Series
from matplotlib import pyplot
from numpy import log
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

from pandas import Series
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot

series = Series.from_csv('shampoo-sales.csv', header=0)

#ACF Shampoo
lag_acf = acf(series)
pyplot.figure()
plot_acf(lag_acf)
pyplot.title('Autocorrelation Function')
pyplot.show()

from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(series)
pyplot.show()

#ARIMA(5,1,0) for SHAMPOO Example 

model = ARIMA(series.values, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())
#
AF = model_fit.forecast()[0]
print('forecast value for t+1=','%f' % AF) 

# Shampoo Example PART II
#
X = series.values
size = int(len(X) * 0.66)
train = X[0:size] # мноежество за .....
test = X[size:len(X)] # множество за тестване
history = [x for x in train]
predictions = list()
for t in range(len(test)):
#print(t)
#if t==(len(test)-1): print(history)
 model = ARIMA(history, order=(5,1,0))
 model_fit = model.fit(disp=0)
 output = model_fit.forecast()
 yhat = output[0]
 predictions.append(yhat)
 obs = test[t]
 history.append(obs)
 print('=%i, predicted=%f, expected(real)=%f' % (size+t,yhat, obs)) 


print(history) 
model = ARIMA(history, order=(5,1,0))
AF1 = model_fit.forecast()[0]
print('forecast value for t+1=','%f' % AF1)

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()