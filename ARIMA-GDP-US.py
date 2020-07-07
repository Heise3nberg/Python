# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 17:44:20 2020
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
from matplotlib import pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
import numpy

data = Series.from_csv("USGDP70-91.csv",header = 0)

plt.plot(data) #There is a trend

#ACF
data_lag_acf = acf(data)
plot_acf(data_lag_acf)

plt.title('Autocorrelation Function')
autocorrelation_plot(data)

#PACF
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf
lag_pacf = pacf(data)
plt.figure()
plot_pacf(lag_pacf)


#Creating the ARIMA(1,1,0) model
model = ARIMA(data.values, order=(1,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
#Plot residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
#Dist of residuals
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

AF = model_fit.forecast()[0]
print('forecast value for t+1=','%f' % AF)

#Forecasting with (1,1,0)
#
X = data.values
size = int(len(X) * 0.66)
train = X[0:size]
test = X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(1,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('=%i, predicted=%f, expected(real)=%f' % (size+t,yhat, obs)) 

model = ARIMA(history, order=(1,1,0))
AF1 = model_fit.forecast()[0]
print('forecast value for t+1=','%f' % AF1) 

#Calculate Error
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


#Creating (1,1,1) Model
model = ARIMA(data.values, order=(1,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())
#Plot residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
#Dist of residuals
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

#Forecasting with (1,1,1)
X = data.values
size = int(len(X) * 0.66)
train = X[0:size]
test = X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(1,1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('=%i, predicted=%f, expected(real)=%f' % (size+t,yhat, obs)) 

model = ARIMA(history, order=(1,1,1))
AF2 = model_fit.forecast()[0]
print('forecast value for t+1=','%f' % AF2) 

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

#Creating (2,1,2) Model
model = ARIMA(data.values, order=(2,1,2))
model_fit = model.fit(disp=0)
print(model_fit.summary())
#Plot residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
#Dist of residuals
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

X = data.values
size = int(len(X) * 0.66)
train = X[0:size]
test = X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(2,1,2))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('=%i, predicted=%f, expected(real)=%f' % (size+t,yhat, obs))

model = ARIMA(history, order=(2,1,2))
AF3 = model_fit.forecast()[0]
print('forecast value for t+1=','%f' % AF3) 

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

#(3,1,2) TEST 
from math import exp
logdata = log(data)

model = ARIMA(data.values, order=(2,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
#Plot residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
#Dist of residuals
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

X = data.values
size = int(len(X) * 0.75)
train = X[0:size]
test = X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(2,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('=%i, predicted=%f, expected(real)=%f' % (size+t,yhat, obs))

model = ARIMA(history, order=(2,1,2))
AF4 = model_fit.forecast()[0]
print('forecast value for t+1=','%f' % AF4) 

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()




