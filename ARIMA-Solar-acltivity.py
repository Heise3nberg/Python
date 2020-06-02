# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 01:06:44 2020

@author: Georgi
"""
from pandas import Series
from matplotlib import pyplot
from numpy import log
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import pacf 
from statsmodels.graphics.tsaplots import plot_pacf 

from pandas import Series
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
import numpy
import statsmodels.api as sm
import pandas as pd

dtap = sm.datasets.sunspots.load_pandas().data

X = dtap.values[:,1]

#plot ACF VISUALISATION
plt.plot(X)

autocorrelation_plot(X)

lag_acf = acf(X)
plt.figure()
plot_acf(lag_acf) 

# plot PACF Partial Autocorrelation
lag_pacf = pacf(X)
plt.figure()
plot_pacf(lag_pacf)

#build ARIMA
model = ARIMA(X,order=(2,0,0))
model_fit = model.fit(disp=0) 
print(model_fit.summary()) 

# plot residual errors 
residuals = DataFrame(model_fit.resid)
residuals.plot()
residuals.plot(kind="kde")
print(residuals.describe())

#Durbin Wastson test for residual correlation
dw = sm.stats.durbin_watson(residuals)
print(dw)

#ACF and PACF for the residuals
res_acf = acf(residuals)
plot_acf(res_acf)

res_pacf = pacf(residuals)
plot_pacf(res_pacf)

#Ljung-Box test for lag autocorrelations != 0
r,q,p = sm.tsa.acf(residuals,qstat = True)
data = numpy.c_[range(1,41),r[1:],q,p]
table = DataFrame(data,columns = ["lag","AC","Q","Prob(>Q)"])

print(table.set_index("lag"))

#Make predictions
predictions = model_fit.predict(start=290, end=len(X)+3, dynamic=False)
print(predictions)

plt.plot(predictions)

#Calculate forecast errors
z=model_fit.predict()
MFE = (X-z).mean() 
MFE

MAE = (numpy.abs((X-z).mean())/z).mean()
print(MAE)














