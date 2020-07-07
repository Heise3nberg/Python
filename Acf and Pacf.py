# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:25:18 2020

@author: Georgi
"""
#ACF1 – ACF for original data 
# ACF1.py
# zoomed-in ACF plot of time series 
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

series = Series.from_csv('AirPassengers.csv', header=0)

pyplot.plot(series)

lag_acf = acf(series)
#lag_acf = acf(series, nlags=20)
pyplot.figure()
plot_acf(lag_acf)
pyplot.title('Autocorrelation Function')
pyplot.show()
#
pyplot.figure()
pyplot.plot(lag_acf)
pyplot.title('Autocorrelation Function')
pyplot.show()
#
pyplot.figure()
plot_acf(series, lags=50)
pyplot.show()
#
from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(series)
pyplot.show() 


#ACF2 – ACF for log data 
# ACF2.py
# zoomed-in ACF plot of time series Log function
from pandas import Series
from matplotlib import pyplot
from numpy import log
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

logseries = log(series)
lag_acf = acf(logseries)
pyplot.figure()
plot_acf(lag_acf)
pyplot.title('Autocorrelation Function')
pyplot.show()
#
#
pyplot.figure()
pyplot.plot(lag_acf)
pyplot.title('Autocorrelation Function')
pyplot.show()
#
pyplot.figure()
plot_acf(series, lags=50)
pyplot.show()
#
#
from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(series)
pyplot.show()


#ACF3 – ACF for Differencing Log Data 
# ACF3.py
# zoomed-in ACF plot of time series Differencing Log
from pandas import Series
from matplotlib import pyplot
from numpy import log
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
series = Series.from_csv('airline-passengers.csv', header=0)

logseries = log(series)
#Differencing Log
X=logseries
diff = list()
for i in range(1, len(X)):
 value = X[i] - X[i - 1]
 diff.append(value)
pyplot.plot(diff)
pyplot.show() #
#lag_acf = acf(diff, nlags=20)
lag_acf = acf(diff)
#
pyplot.figure()
plot_acf(lag_acf)
pyplot.title('Autocorrelation Function')
pyplot.show()
#
#
pyplot.figure()
pyplot.plot(lag_acf)
pyplot.title('Autocorrelation Function')
pyplot.show()
#
pyplot.figure()
plot_acf(diff, lags=20)
pyplot.show()
#
pyplot.figure(dpi=80, figsize=(14, 7))
plot_acf(diff) #, lags=20)
pyplot.show()
#
#

from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(diff)
pyplot.show()

#PACF1 – PACF for original data
# PACF1.py
# zoomed-in PACF plot of time series
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf

lag_pacf = pacf(series)

#lag_acf = acf(series, nlags=20)
print(' ORIGINAL DATA')
pyplot.figure()
plot_pacf(lag_pacf)
pyplot.title('Partial Autocorrelation Function')
pyplot.show()
#
pyplot.figure()
pyplot.plot(lag_pacf)
pyplot.title('Partial Autocorrelation Function')
pyplot.show()
#
pyplot.figure()
plot_pacf(series, lags=50)
pyplot


# PACF2 – PACF for log data
# PACF2.py 
# zoomed-in PACF plot of time series Log function
from pandas import Series
from matplotlib import pyplot
from numpy import log
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf

logseries = log(series)
lag_pacf = pacf(logseries)
print(' LOG ORIGINAL DATA')
pyplot.figure()
plot_pacf(lag_pacf)
pyplot.title('Partial Autocorrelation Function')
pyplot.show()
#
#
pyplot.figure()
pyplot.plot(lag_pacf)
pyplot.title('Partial Autocorrelation Function')
pyplot.show()
#
pyplot.figure()
plot_pacf(series, lags=50)
pyplot.show()


# PACF3 – PACF for Differencing Log Data
# PACF3
# zoomed-in PACF plot of time series Differencing Log function
from pandas import Series
from matplotlib import pyplot
from numpy import log
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf

logseries = log(series)
#Differencing Log
X=series
diff = list()
for i in range(1, len(X)):
 value = X[i] - X[i - 1]
 diff.append(value)
print(' Differencing LOG ORIGINAL DATA')
pyplot.plot(diff)
pyplot.show()
#
#lag_pacf = pacf(diff, nlags=20, method='ols')
lag_pacf = pacf(diff)
#
pyplot.figure()
plot_pacf(lag_pacf)
pyplot.title('Partial Autocorrelation Function')
pyplot.show()
#
#
pyplot.figure()
pyplot.plot(lag_pacf)
pyplot.title('Partial Autocorrelation Function')
pyplot.show()
#
pyplot.figure()
plot_pacf(series, lags=50)
pyplot.show()















