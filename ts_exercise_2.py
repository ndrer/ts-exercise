import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame

start = datetime.datetime(2020, 1, 1)
end = datetime.datetime.now()

df = web.DataReader("UNVR.JK", "yahoo", start, end)
print(df.tail(5))

print(df.shape)

close_px = df['Adj Close']
mavg = close_px.rolling(window = 100).mean()
mavg.tail(10)

# Add %matplotlib inline if necessary
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl

mpl.rc('figure', figsize = (15, 7))
mpl.__version__

style.use('ggplot')
close_px.plot(label = 'UNVR.JK')
mavg.plot(label = 'mavg')
plt.legend()

rets = close_px / close_px.shift(1) - 1
rets.head(5)

rets.plot(label = 'return')

dfcomp = web.DataReader(['TSPC.JK', 'ULTJ.JK', 'MYOR.JK', 'ICBP.JK', 'UNVR.JK'], 'yahoo', start = start, end = end)['Adj Close']
dfcomp.columns = dfcomp.columns.str.replace('.JK', '')

print(dfcomp.tail(5))

print(dfcomp.shape)

retscomp = round(dfcomp.pct_change(), 4)
corr = retscomp.corr()

print(corr)

plt.scatter(retscomp.ICBP, retscomp.UNVR)
plt.xlabel('Returns ICBP')
plt.ylabel('Returns UNVR')

from pandas.plotting import scatter_matrix
scatter_matrix(retscomp, diagonal = 'kde', figsize = (10, 10))
plt.show();


plt.imshow(corr, cmap = None, interpolation = 'none',vmin=0, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns);

# Add plt show because something wrong with the scaling
plt.show()

plt.scatter(retscomp.mean(), retscomp.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
  plt.annotate(
      label,
      xy = (x, y ), xytext = (20, -20),
      textcoords = 'offset points', ha = 'right', va = 'bottom',
      bbox = dict(boxstyle = 'round,pad = 0.5', fc = 'yellow', alpha = 0.5), 
      arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
  )
plt.show()

dfreg = df.loc[:,['Adj Close', 'Volume']]
dfreg['HiLo_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['Delta_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
print(dfreg.head())

# Table for quadratic model
dfreg1 = df.loc[:,['Adj Close', 'Volume']]
dfreg1['HiLo_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg1['Delta_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

# Table for KNN model
dfreg2 = df.loc[:,['Adj Close', 'Volume']]
dfreg2['HiLo_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg2['Delta_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate

# Change NA to -99999, should be dropped later
dfreg.fillna(value = -99999, inplace = True)

print(dfreg.shape)

# Separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))

# Separating the label, predict the Adj_Close
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'],1))

# Scale the X for linreg
X = preprocessing.scale(X)

# Late X and early X train data
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

#Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

print('Dimension of X', X.shape)
print('Dimension of y', y.shape)

# Quadratic model

# Similar procedure to above

dfreg1.fillna(value = -99999, inplace = True)
forecast_out1 = int(math.ceil(0.01 * len(dfreg1)))

forecast_col1 = 'Adj Close'
dfreg1['label'] = dfreg1[forecast_col1].shift(-forecast_out1)
X1 = np.array(dfreg1.drop(['label'],1))

X1 = preprocessing.scale(X1)

X_lately1 = X1[-forecast_out1:]
X1 = X1[:-forecast_out1]

y1 = np.array(dfreg1['label'])
y1 = y1[:-forecast_out1]

# KNN model

# Similar procedure to above
dfreg2.fillna(value = -99999, inplace = True)

forecast_out2 = int(math.ceil(0.01 * len(dfreg2)))

forecast_col2 = 'Adj Close'
dfreg2['label'] = dfreg2[forecast_col2].shift(-forecast_out2)
X2 = np.array(dfreg2.drop(['label'],1))

X2 = preprocessing.scale(X2)

X_lately2 = X2[-forecast_out2:]
X2 = X2[:-forecast_out2]

y2 = np.array(dfreg2['label'])
y2 = y2[:-forecast_out2]

# Train test splitting

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.2)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.2)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Linear regression
clfreg = LinearRegression(n_jobs = -1)
clfreg.fit(X_train, y_train)

# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X1_train, y1_train)


# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors = 2)
clfknn.fit(X2_train, y2_train)

confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test, y_test)
confidencepoly3 = clfpoly3.score(X1_test, y1_test)
confidenceknn = clfknn.score(X2_test, y2_test)

print("The linear regression confidence is", confidencereg)
print("The quadratic regression 2 confidence is ", confidencepoly2)
print("The quadratic regression 3 confidence is ", confidencepoly3)
print("The knn regression confidence is ", confidenceknn)

forecast_set = clfreg.predict(X_lately)
dfreg['Forecast'] = np.nan
print(forecast_set, confidencereg, forecast_out)

forecast_set1 = clfpoly3.predict(X_lately)
dfreg1['Forecast'] = np.nan
print(forecast_set1, confidencepoly3, forecast_out1)

forecast_set2 = clfknn.predict(X_lately)
dfreg2['Forecast'] = np.nan
print(forecast_set2, confidenceknn, forecast_out2)

last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days = 1)

for i in forecast_set:
  next_date = next_unix
  next_unix += datetime.timedelta(days=1)
  dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]

dfreg['Adj Close'].tail(120).plot()
dfreg['Forecast'].tail(120).plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Linear Regression Model Predictions of UNVR Prices')
plt.show()

last_date = dfreg1.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days = 1)

for i in forecast_set1:
  next_date = next_unix
  next_unix += datetime.timedelta(days=1)
  dfreg1.loc[next_date] = [np.nan for _ in range(len(dfreg1.columns)-1)]+[i]

plt.figure(figsize=(10,3))
dfreg1['Adj Close'].tail(120).plot()
dfreg1['Forecast'].tail(120).plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Quadratic Reg Pred of UNVR Prices')
plt.show()

last_date = dfreg2.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days = 1)

for i in forecast_set2:
  next_date = next_unix
  next_unix += datetime.timedelta(days=1)
  dfreg2.loc[next_date] = [np.nan for _ in range(len(dfreg2.columns)-1)]+[i]

plt.figure(figsize=(10,3))
dfreg2['Adj Close'].tail(120).plot()
dfreg2['Forecast'].tail(120).plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('KNN Pred of UNVR Prices')
plt.show()