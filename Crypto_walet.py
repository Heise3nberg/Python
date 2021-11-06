# -*- coding: utf-8 -*-

import pandas as pd
import pandas_datareader
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import plot,iplot
from plotly.subplots import make_subplots
import streamlit as st

st.write('''#Inverstment portfolio analysis''')

# %%
#please choose portfolio tickers:
tickers = ['AAPL','TSLA','GOOGL','AMZN','IBM','NVDA']

#choose the start date:
start = dt.datetime(2020,1,31)

#we assume that the end date is TODAY:
end = dt.datetime.now()

# %%
#import the data
#create a list of the series
prices = []

for i in tickers:
    p = pandas_datareader.DataReader(i,
                                    data_source='yahoo',
                                    start=start,
                                    end = end)['Adj Close']
    prices.append(p)

# %%
#some manual checks to see if the prices consistent:
prices[0].index.difference(prices[1].index) #do not match
prices[0].index.difference(prices[2].index) #do not match
prices[0].index.difference(prices[3].index) #match

#create a sequence of dates to be used for refference
date_seq = pd.date_range(start=start, end=end)

#create a list to hold the missing dates for each crypto
dates = [[] for l in range(len(prices))] #reverse the list of dates if needed with: "last_first = [date for date in dates[::-1]]"

#find the missing dates
for date in date_seq:
        for p in range(0,len(prices)):          
            if date in prices[p].index:
                dates[p].append(date)
            else:
                dates[p].append('missing')

#check differences
for i in range(0,len(dates)):
    print(date_seq.difference(dates[i]))

#Imporve this part to work with any N number of tickers, now it works with only 4
#get missing dates
missing_l = [[] for l in range(len(dates))] 

for i in range(0,len(dates)):
    missing_l[i] = date_seq.difference(dates[i])

for i in range(0,len(missing_l)):
    plt.bar(missing_l[i],len(missing_l[i]),label=tickers[i])
plt.legend(loc='best')

# %%

#create date mapping
prices_new = pd.DataFrame(columns = ["Date"])
prices_new['Date'] = pd.to_datetime(date_seq)

#do a duplicate check
for i in range (0,len(prices)):
    print(sum(prices[i].index.duplicated())) #there are duplicated rows in each series

#merge and remove duplicates
for p in prices:
    prices_new = pd.merge(prices_new, p,  on="Date", how="left")
    if sum(prices_new['Date'].duplicated()) > 0:
        prices_new = prices_new.drop_duplicates(subset = "Date")

#duplicates check
prices_new['Date'][prices_new['Date'].duplicated()]
print('Duplicated dates:',sum(prices_new['Date'].duplicated()))

# %%
#rearrange dataframe
prices_new = prices_new.set_index("Date")
prices_new.columns = tickers

#count the N of missing values
prices_new.isna().sum()

#impute missing data
#Imputation: 
   # Because the series are not missing due to random mistakes, 
       # we can umpute the with the lagged variable a.k.a (LOCF):
prices_new.fillna(method="ffill", inplace=True)

#cum returns and vol
vol = prices_new.pct_change(1).std()
r = prices_new.pct_change(1).mean()

#function for returns
def returns(x):
    return (x - x.shift(1))/x.shift(1)

#get new dataframe with returns
rets = prices_new.apply(returns)

#compare function with pct_change
rets.mean()
rets.std()

#plot absolute values
fig = px.line(prices_new, x=prices_new.index, y=prices_new.columns)
plot(fig)

#prices_new.plot()

# %%
#plot log values
fig = px.line(np.log(prices_new),x=prices_new.index, y=prices_new.columns)
plot(fig)

# %%
#plot returns (create subplots)
# prices_new.pct_change(1).plot()
fig = px.line(rets,x=rets.index, y=rets.columns)
plot(fig)

# total = len(rets.columns)
# col_len = math.ceil(len(rets.columns)/2)
# row_len = total // col_len
# row_len += total % col_len

# fig = make_subplots(rows=col_len,cols=row_len)
# for i in range(1,row_len):
#     for j in range(1,col_len):
#         for k in range(total):
#             trace = go.Scatter(x=rets.iloc[:,[k]],name=rets.columns[k])
#             fig.append_trace(trace,i,j)
         
# fig.show()  
    
# %%
#plot correlations
plt.figure()
prices_new.pct_change().corr()
sns.heatmap(prices_new.pct_change().corr(),annot = True)
plt.title('Correlations')

# %%
#create automated plots
import matplotlib.colors as mcolors

total = len(rets.columns)
col_len = math.ceil(len(rets.columns)/2)
row_len = total // col_len
row_len += total % col_len

position = range(1,total+1)

plt.figure()
fig1 = plt.figure(1)
for k in range(total):

  ax1 = fig1.add_subplot(col_len,row_len,position[k])
  fig1.suptitle('Histogram of returns')
  ax1.hist(rets.iloc[:,[k]],color = list(mcolors.TABLEAU_COLORS.keys())[k])
  ax1.set_xlabel(rets.columns[k])
 
# %%
#volatility vs returns over time
time_window = 30

monthly_rets = rets.rolling(window=30).mean()
monthly_vol = rets.rolling(window=30).std()

plt.figure()
fig2 = plt.gcf()
for k in range(total):

  ax2 = fig2.add_subplot(col_len,row_len,position[k])
  fig2.suptitle('Monthly returns/volatility')
  ax2.plot(monthly_rets.iloc[:,[k]],label='returns')
  ax2.plot(monthly_vol.iloc[:,[k]],label='volatility')
  ax2.set_xlabel(monthly_rets.columns[k])
  ax2.legend()
   
# %%
#Sortino and sharpre ratio
#cumulative volatility and expected returns
r = prices_new.pct_change(1).mean() #expected returns
s_volatility = rets[rets < 0].std() #vol by sortino

Sortino_ratio = r/s_volatility

#Create sortino plot
plt.figure()
Sortino_ratio.plot(kind='bar',color=list(mcolors.TABLEAU_COLORS.keys())[0:len(r)])
plt.yscale('symlog')
plt.title('Sortino ratio')

#just in case



