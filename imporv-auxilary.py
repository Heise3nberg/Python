# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:21:08 2020

@author: Georgi
"""

size = int(len(X) * 0.66)
train = X[0:size]
test = X[size:len(X)]


btest = DataFrame(residuals)
btest.reshape(-1, 1)
btrain = DataFrame(residuals)
btrain.reshape(-1, 1)


#MA(0,2,3) for O 

O_model = ARIMA(O, order=(0,2,3))
O_model_fit = O_model.fit(disp=0)
print(O_model_fit.summary())

O_size = int(len(O) * 0.66)
O_train = O[0:O_size]
O_test = O[O_size:len(O)]
O_history = [x for x in O_train]
O_predictions = list()

for t in range(len(O_test)):
   O_model = ARIMA(O_history, order=(0,2,3))
   O_model_fit = O_model.fit(disp=0)
   output = O_model_fit.forecast()
   Ohat = output[0]
   O_predictions.append(Ohat)
   obs = O_test[t]
   O_history.append(obs)
   print('=%i, predicted=%f, expected(real)=%f' % (O_size+t,Ohat, obs)) 

print(O_predictions)

O_final = np.array(O_predictions)

#AR (0,1,3) for h

acf_h = acf(h)
plot_acf(acf_h)

pacf_h = pacf(h)
plot_pacf(pacf_h)

h_model = ARIMA(h, order=(1,0,0))
h_model_fit = h_model.fit(disp=0)
print(h_model_fit.summary())

h_size = int(len(h) * 0.66)
h_train = h[0:h_size]
h_test = h[h_size:len(h)]
h_history = [x for x in h_train]
h_predictions = list()

for t in range(len(h_test)):
   h_model = ARIMA(h_history, order=(1,0,0))
   h_model_fit = h_model.fit(disp=0)
   output = h_model_fit.forecast()
   hhat = output[0]
   h_predictions.append(hhat)
   obs = h_test[t]
   h_history.append(obs)
   print('=%i, predicted=%f, expected(real)=%f' % (h_size+t,hhat, obs)) 

a = np.array(h_predictions)

b = abs(h - a)



svr = SVR(kernel='poly', C=1, gamma='auto', epsilon = 0.2)



y_poly = svr.fit(btrain,btest).predict(btrain)

b_predictions = svr.predict(btrain)

plt.plot(b_predictions)

plt.plot(btrain, y_poly, c='b', label='Polynomial model')


b_final = b_predictions[280:]

o_final = o_predictions[:29]

a_final = h_predictions[:29]

forecast = b_final + o_final + a_final

plt.plot(forecast)

plt.plot(series)