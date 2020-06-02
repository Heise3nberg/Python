# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:55:13 2020

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

plt.plot(series)

#CREATE EXPONENTIAL MOVING AVERAGE
#y_t = a * y_t + a * (1-a)^1 * y_t-1 + a * (1-a)^2 * y_t-2

from statsmodels.tsa.api import ExponentialSmoothing

exp = ExponentialSmoothing(series)

exp_model = exp.fit(smoothing_level = 0.1) 

o = exp_model.fittedvalues

#DEDUCE HIGH VARIANCE SERIES

h = (series - o)

plt.plot(o)
plt.plot(h)

#MEASURE ACF AND PACF FOR o(low variance)
acf_o = acf(o)
plot_acf(acf_o)

pacf_o = pacf(o)
plot_pacf(pacf_o)

#FIT ARIMA Model for o 
o_model = ARIMA(o,order=(1,0,4))
o_model_fit = o_model.fit(disp=0) 
print(o_model_fit.summary())

osize = int(len(o)*0.66) 
otrain = o[0:osize] 
otest = o[osize:len(o)]

ohistory = [x for x in otrain]
opredictions = list()

for t in range(len(otest)):
    o_model = ARIMA(ohistory,order=(1,0,4))
    o_model_fit = o_model.fit(disp=0) 
    output = o_model_fit.forecast()
    ohat = output[0]
    opredictions.append(ohat)
    obs = otest[t]
    ohistory.append(obs)

mean_squared_error(otest, opredictions)

plt.plot(otest)
plt.plot(opredictions)

#ACF and PACF for h (high variance)

acf_h = acf(h)
plot_acf(acf_h)

pacf_h = pacf(h)
plot_pacf(pacf_h)

#FIT AR MODEL FOR h

h_model = ARIMA(h,order=(5,0,0))
h_model_fit = h_model.fit(disp=0) 
print(h_model_fit.summary())

hsize = int(len(h)*0.5) 
htrain = h[0:hsize] 
htest = h[hsize:len(h)]

hhistory = [x for x in htrain]
hpredictions = list()

for t in range(len(htest)):
    h_model = ARIMA(hhistory,order=(5,0,0))
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
hpredictions = np.array(hpredictions)
hpredictions = hpredictions.flatten()

newb = (htest - hpredictions)

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

newbtrain,newbtest = train_test_split(newb,test_size = 0.33)

newbtrain = DataFrame(newbtrain)
newbtest = DataFrame(newbtest)

svr = SVR(kernel='poly', C=0.1, epsilon = 0.07)

newb_poly = svr.fit(newbtrain,newbtrain)

newb_predictions = svr.predict(newbtest)

plt.plot(newb_predictions)

plt.plot(newbtest, newb_predictions, c='b', label='Polynomial model')


#CREATE FINAL FORECAST
b_final = newb_predictions

o_final = opredictions[103:]

a_final = hpredictions[103:]


o_final = np.array(o_final)
o_final = o_final.flatten()

forecast = b_final + o_final + a_final
plt.plot(forecast)

conc = series[:257]
forecast1 = np.concatenate([conc,forecast])

plt.plot(series)
plt.plot(forecast1)

error = mean_squared_error(forecast1,series)
print('Test MSE: %.3f' % error)




















