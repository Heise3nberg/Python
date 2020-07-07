# -*- coding: utf-8 -*-
"""
Created on Sun May  3 23:58:30 2020

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
import statsmodels.api as sm
import pandas as pd
from numpy import sqrt
from pandas import DataFrame
from sklearn.metrics import mean_squared_error

data = sm.datasets.sunspots.load_pandas().data

series = data.values[:,1]

X = data.values[:,1]

plt.plot(series)


#CREATE EXPONENTIAL MOVING AVERAGE
#y_t = a * y_t + a * (1-a)^1 * y_t-1 + a * (1-a)^2 * y_t-2

from statsmodels.tsa.api import ExponentialSmoothing

exp = ExponentialSmoothing(series)

exp_model = exp.fit(smoothing_level = 0.2) 

o = exp_model.fittedvalues

#DEDUCE LOW AND HIGH VARIANCE SERIES

h = (series - o)

plt.plot(o)
plt.plot(h)

print(h)

#ACF AND PACF FOR O
acf_o = acf(o)
plot_acf(acf_o)

pacf_o = pacf(o)
plot_pacf(pacf_o)


#Model ARIMA(1,0,3) for o 

o_model = ARIMA(o,order=(1,1,3))
o_model_fit = o_model.fit(disp=0) 
print(o_model_fit.summary())

o_predictions = o_model_fit.predict(start=2, end=len(o)+2, dynamic=False)
print(o_predictions)

plt.plot(o_predictions)



osize = int(len(o)*0.5) 
otrain = o[0:osize] 
otest = o[osize:len(o)]

ohistory = [x for x in otrain]
opredictions = list()

for t in range(len(otest)):
    o_model = ARIMA(ohistory,order=(1,1,3))
    o_model_fit = o_model.fit(disp=0) 
    output = o_model_fit.forecast()
    ohat = output[0]
    opredictions.append(ohat)
    obs = otest[t]
    ohistory.append(obs)

mean_squared_error(otest, opredictions)

plt.plot(otest)
plt.plot(opredictions)

#ACF and PACF for AR model

acf_h = acf(h)
plot_acf(acf_h)

pacf_h = pacf(h)
plot_pacf(pacf_h)


from statsmodels.tsa.stattools import adfuller
h_adfuller = adfuller(h)
print(h_adfuller)

#Model AR (2,0,0) for h

h_model = ARIMA(h,order=(2,0,0))
h_model_fit = h_model.fit(disp=0) 
print(h_model_fit.summary())

h_predictions = h_model_fit.predict(start=0, end=len(h)-1, dynamic=False)
print(h_predictions)

plt.plot(h_predictions)

print(h_predictions.shape)

residuals = DataFrame(h_model_fit.resid)
residuals.plot()



hsize = int(len(h)*0.5) 
htrain = h[0:hsize] 
htest = h[hsize:len(h)]

hhistory = [x for x in htrain]
hpredictions = list()

for t in range(len(htest)):
    h_model = ARIMA(hhistory,order=(2,0,0))
    h_model_fit = h_model.fit(disp=0) 
    output = h_model_fit.forecast()
    hhat = output[0]
    hpredictions.append(hhat)
    obs = htest[t]
    hhistory.append(obs)

mean_squared_error(htest, hpredictions)

plt.plot(htest)
plt.plot(hpredictions)

#DEDUCE ERROR FROM HIGH-VOLATILITY AR MODEL
b = (h - h_predictions)

#CREATE SUPPORT VECTOR REGRESSION MODEL
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

btrain,btest = train_test_split(b,test_size = 0.33)

btrain = DataFrame(btrain)
btest = DataFrame(btest)

svr = SVR(kernel='poly', C=0.1, epsilon = 0.07)

b_poly = svr.fit(btrain,btrain)

b_predictions = svr.predict(btest)

print(b_predictions)
plt.plot(b_predictions)

plt.plot(btest, b_predictions, c='b', label='Polynomial model')

#DEDUCE ERROR FROM HIGH-VOLATILITY AR MODEL NEWWWWWWWWWWWWWWWWWWWWWWWWWW
hpredictions = np.array(hpredictions)
hpredictions = hpredictions.flatten()

newb = (htest - hpredictions)

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

newbtrain,newbtest = train_test_split(newb,test_size = 0.33)

newbtrain = DataFrame(newbtrain)
newbtest = DataFrame(newbtest)



newb_poly = svr.fit(newbtrain,newbtrain)

newb_predictions = svr.predict(newbtest)

plt.plot(newb_predictions)

plt.plot(newbtest, newb_predictions, c='b', label='Polynomial model')



b_final = newb_predictions

o_final = opredictions[103:]

a_final = hpredictions[103:]

o_final = np.array(o_final)
o_final = o_final.flatten()

forecast = b_final + o_final + a_final

conc = series[:257]
forecast1 = np.concatenate([conc,forecast])

plt.plot(series)
plt.plot(forecast1)

error = mean_squared_error(forecast1,series)
print('Test MSE: %.3f' % error)














