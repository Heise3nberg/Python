# -*- coding: utf-8 -*-
"""
Created on Sun May 10 23:49:56 2020

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

euro = Series.from_csv("euro-dollar.csv",header = 0, index_col=0)

plt.plot(euro)

euro = euro[1260:1600]

plt.figure(figsize=(10,4))
plt.plot(euro)
#stationarity check

from statsmodels.tsa.stattools import adfuller

adfuller = adfuller(euro)

print(adfuller)
print('ADF Statistic: %f' % adfuller[0])
print('p-value: %f' % adfuller[1])
print('Critical Values:')
for key, value in adfuller[4].items():
        print('\t%s: %.3f' % (key, value))
        
        
#Series in not stationary
from statsmodels.tsa.stattools import acf
acf = acf(euro)
plot_acf(acf,lags=30)

from statsmodels.tsa.stattools import pacf
pacf = pacf(euro)
plot_pacf(pacf,lags=30)

#Examine differenced series
diff = list()
for i in range(1, len(euro)):
 value = euro[i] - euro[i - 1]
 diff.append(value)

first_order_diff = euro.diff(1)

diff = DataFrame(diff)
plt.plot(diff)

from statsmodels.tsa.stattools import acf
acf_diff = acf(diff)
plot_acf(acf_diff)


#Begin to fit model

model = ARIMA(euro.values, order=(2,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

forecast = model_fit.forecast()
print(forecast)

#Plot residuals 

residuals = DataFrame(model_fit.resid)
residuals.plot()
residuals.plot(kind='kde')
print(residuals.describe())

#Test residuals
DW = sm.stats.durbin_watson(residuals)
print(DW)

#ACF nad PACF of residuals
from statsmodels.tsa.stattools import acf

res_acf = acf(residuals)
plot_acf(res_acf,lags=30)

from statsmodels.tsa.stattools import pacf
res_pacf = pacf(residuals)
plot_pacf(res_pacf,lags=30)

# lag, autocorrelation (AC), Q statistic and Prob>Q.

r,q,p = sm.tsa.acf(residuals, fft=True, qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))

from math import exp
#make predictions
X=euro.values

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



#MSE
from sklearn.metrics import mean_squared_error
error = mean_squared_error(test, predictions)
print('Test MSE=', error)

#MFE
MFE = (predictions-test).mean()
print("MFE = ",MFE)

#MAE
MAE = (abs((predictions-test).mean()) / predictions).mean()
print("MAE = ", MAE)






















