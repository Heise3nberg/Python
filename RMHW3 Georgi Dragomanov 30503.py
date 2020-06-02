# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:45:33 2020

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

from scipy.stats import norm
import pandas_datareader
from pandas_datareader import data
from math import sqrt
from statsmodels.tsa.stattools import adfuller

stocks = data.DataReader('BA', 'yahoo',start='1/1/2009',end='1/1/2014')

plt.plot(stocks["Adj Close"])

series = stocks["Adj Close"].values

#ACF/PACF
plot_acf(acf(series))

plot_pacf(pacf(series))

#1)OLIVEIRA LUDERMIR ALGORITHM
#Exp Smoothing

from statsmodels.tsa.api import ExponentialSmoothing

exp = ExponentialSmoothing(series)

exp_model = exp.fit(smoothing_level=0.2)

o = exp_model.fittedvalues


#Deducing high/low variance
h = (series - o)

plt.plot(h)
plt.plot(o)

#ACF ANALYSIS FOR H AND O

acf_o = acf(o)
plot_acf(acf_o)

pacf_o = pacf(o)
plot_pacf(pacf_o)

from statsmodels.tsa.stattools import adfuller

adfullero = adfuller(o)
print(adfullero)
print('ADF Statistic: %f' % adfullero[0])
print('p-value: %f' % adfullero[1])
print('Critical Values:')
for key, value in adfullero[4].items():
        print('\t%s: %.3f' % (key, value))

#Not stationary
#############################


diffo = list()
for i in range(1, len(o)):
 value = o[i] - o[i - 1]
 diffo.append(value)

acf_diff = acf(diffo)
plot_acf(acf_diff)

pacf_diff = pacf(diffo)
plot_pacf(pacf_diff)

#After 1st differencing we look at a stationary process:
from statsmodels.tsa.stattools import adfuller
adfuller_o = adfuller(diffo)
print(adfuller_o)
print('ADF Statistic: %f' % adfuller_o[0])
print('p-value: %f' % adfuller_o[1])
print('Critical Values:')
for key, value in adfuller_o[4].items():
        print('\t%s: %.3f' % (key, value))

#starting ARIMA parameters for o will be: 3,1,1

#FIT ARIMA Model for o 
o_model = ARIMA(o,order=(2,1,1))
o_model_fit = o_model.fit(disp=0) 
print(o_model_fit.summary())
#Evaluate AIC and pick a model: 2,1,1 show best AIC value

#Check residuals
plt.plot(o_model_fit.resid)

residualso = DataFrame(o_model_fit.resid)

residualso.plot(kind='kde')
print(residualso.describe())

DW = sm.stats.durbin_watson(residualso)
print('DW=', DW)



osize = int(len(o)*0.66) 
otrain = o[0:osize] 
otest = o[osize:len(o)]

ohistory = [x for x in otrain]
opredictions = list()

for t in range(len(otest)):
    o_model = ARIMA(ohistory,order=(2,1,1))
    o_model_fit = o_model.fit(disp=0) 
    output = o_model_fit.forecast()
    ohat = output[0]
    opredictions.append(ohat)
    obs = otest[t]
    ohistory.append(obs)

mean_squared_error(otest, opredictions)

plt.plot(otest)
plt.plot(opredictions)


#######FIT AR MODEL FOR h
#AD FULLER and other testing
from statsmodels.tsa.stattools import adfuller
adfullerh = adfuller(h)
print(adfullerh)
print('ADF Statistic: %f' % adfullerh[0])
print('p-value: %f' % adfullerh[1])
print('Critical Values:')
for key, value in adfullerh[4].items():
        print('\t%s: %.3f' % (key, value))

acf_h = acf(h)
plot_acf(acf_h)

pacf_h = pacf(h)
plot_pacf(pacf_h)

#####################################################
h_model = ARIMA(h,order=(1,0,0))
h_model_fit = h_model.fit(disp=0) 
print(h_model_fit.summary())

hsize = int(len(h)*0.66) 
htrain = h[0:hsize] 
htest = h[hsize:len(h)]

hhistory = [x for x in htrain]
hpredictions = list()

for t in range(len(htest)):
    h_model = ARIMA(hhistory,order=(1,0,0))
    h_model_fit = h_model.fit(disp=0) 
    output = h_model_fit.forecast()
    hhat = output[0]
    hpredictions.append(hhat)
    obs = htest[t]
    hhistory.append(obs)

mean_squared_error(htest, hpredictions)

plt.plot(htest)
plt.plot(hpredictions)


#DEDUCE ERROR FROM THE HIGH-VOLATILITY AR MODEL
hpredictions = np.array(hpredictions)
hpredictions = hpredictions.flatten()

b = (htest - hpredictions)

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

btrain,btest = train_test_split(b,test_size = 0.33)

btrain.reshape(-1,1)
btest.reshape(-1,1)

btrain = np.array(btrain)
btest = np.array(btest)

svr = SVR(kernel='rbf', C=1, epsilon = 0.05)

bpoly = svr.fit(btrain.reshape(-1,1),btrain.reshape(-1,1))

b_predictions = svr.predict(btest.reshape(-1,1))

plt.plot(btest)
plt.plot(b_predictions)

#CREATE FINAL FORECAST

b_final = b_predictions

a_final = hpredictions[220:]

o_final = opredictions[220:]

o_final = np.array(o_final)
o_final = o_final.flatten()

a_final = np.array(a_final)
a_final = a_final.flatten()

forecast = a_final + b_final + o_final

plt.plot(forecast)

#Compare results
lenght = len(series) - len(forecast)

print(lenght)

testvalues = series[lenght:]

#Results Visualization:
plt.plot(testvalues)
plt.plot(forecast)

#Calculate error:
error = mean_squared_error(forecast,testvalues)
print('Test RMSE: %.3f' % error)

smape = 0
for i in range(len(testvalues)):
 x=(abs(testvalues[i])+abs(forecast[i]))/2
 smape += abs(testvalues[i]-forecast[i])/x

smape = smape/(len(testvalues)+1)

print('ARIMA SMAPE: %f' % smape)

mae = 0
for i in range(len(testvalues)):
 mae += abs(testvalues[i]-forecast[i])

mae = mae/(len(testvalues))

print(' ARIMA for Finance time Series MAE: %f' %mae)




#2) Алгоритъм ARIMA 1

Y = series
Y = np.array(Y).astype('float')
n = len(Y)
print('Number of observations =', len(Y))

plot_acf(acf(Y))

plot_pacf(pacf(Y))

diff1 = list()
for i in range(1, len(Y)):
 value = Y[i] - Y[i - 1]
 diff1.append(value)

acf_diff1 = acf(diff1)
plot_acf(acf_diff1)

pacf_diff1 = pacf(diff1)
plot_pacf(pacf_diff1)

order1 = (2,1,0)

size1 = int(len(Y)*0.7)
train1 = Y[0:size1]
test1 = Y[size1:len(Y)]

history1 = [x for x in train1]
predictions1 = list()

for t in range(len(test1)):
 model1 = ARIMA(history1, order1)
 model_fit1 = model1.fit(disp=0)
 output = model_fit1.forecast()
 yhat = output[0]
 predictions1.append(yhat)
 obs = test1[t]
 history1.append(obs)
 print('=%i, predicted=%f, expected(real)=%f' % (size1+t,yhat, obs)) 

#Check residuals
residuals1 = DataFrame(model_fit1.resid)

residuals1.plot()

residuals1.plot(kind='kde')
print(residuals1.describe())

DW1 = sm.stats.durbin_watson(residuals1)
print('DW=', DW1)

#Check errors:

error1 = mean_squared_error(test1, predictions1)
print('Test RMSE: %.3f' % error1)

smape1 = 0
for i in range(len(test1)):
 x=(abs(test1[i])+abs(predictions1[i]))/2
 smape1 += abs(test1[i]-predictions1[i])/x

smape1 = smape1/(len(test1)+1)

print('ARIMA SMAPE: %f' % smape)

mae1 = 0
for i in range(len(test1)):
 mae1 += abs(test1[i]-predictions1[i])
mae1 = mae/(len(test1))

print(' ARIMA for Finance time Series MAE: %f' %mae1)




#2) Алгоритъм ARIMA 2

order2 = (2,1,0)

size2 = int(len(Y)*0.7)
train2 = Y[0:size2]
test2 = Y[size2:len(Y)]

model2 = ARIMA(train2, order2)
model_fit2 = model2.fit(disp=0)
print(model_fit2.summary())

start = size2

end = len(Y)-1

pred2 = model_fit2.predict(start, end, dynamic=True)
forecast2 = model_fit2.forecast(steps=len(test2))[0]


error2 = mean_squared_error(test2, forecast2)
print('Test RMSE: %.3f' % error2)

smape2 = 0

for i in range(len(test2)):
 a=(abs(test2[i])+abs(pred2[i]))/2
 smape2 += abs(test2[i]-pred2[i])/a

smape2 = smape2/(len(test2)+1)

print('ARIMA for Finance time Series SMAPE: %f ' %smape2)

for i in range(len(test2)-3,len(test2)):
 print('=%i, expected(real)=%f, predicted=%f , forcasted=%f'
 % (i,test2[i], pred2[i], forecast2[i]))

mae2 = 0

for i in range(len(test2)):
 mae2 += abs(test2[i]-pred2[i])

mae2 = mae2/(len(test2)) 

print(' ARIMA for Finance time Series MAE: %f' %mae2)

stocks = pd.DataFrame(stocks)
pd.DataFrame.to_csv(r'C:\Users\Georgi\Desktop\Data science\rabotna - Python\stocks.csv')












