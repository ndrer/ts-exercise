# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 13:08:04 2021

@author: Andreas
"""

#Import library
# add %matplotlib inline if using notebook
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np

from statsmodels.tsa.arima.model import ARIMA

from sklearn.utils.validation import check_consistent_length, check_array

# Defining MAPE function since sklearn 0.24 is not available on conda
# Delete this one if sklearn 0.24 is available

def mean_absolute_percentage_error(y_true, y_pred,
                                   sample_weight=None,
                                   multioutput='uniform_average'):
    
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    output_errors = np.average(mape,
                               weights=sample_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)

def _check_reg_targets(y_true, y_pred, multioutput, dtype="numeric"):
    
    check_consistent_length(y_true, y_pred)
    y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
    y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError("y_true and y_pred have different number of output "
                         "({0}!={1})".format(y_true.shape[1], y_pred.shape[1]))

    n_outputs = y_true.shape[1]
    allowed_multioutput_str = ('raw_values', 'uniform_average',
                               'variance_weighted')
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError("Allowed 'multioutput' string values are {}. "
                             "You provided multioutput={!r}".format(
                                 allowed_multioutput_str,
                                 multioutput))
    elif multioutput is not None:
        multioutput = check_array(multioutput, ensure_2d=False)
        if n_outputs == 1:
            raise ValueError("Custom weights are useful only in "
                             "multi-output cases.")
        elif n_outputs != len(multioutput):
            raise ValueError(("There must be equally many custom weights "
                              "(%d) as outputs (%d).") %
                             (len(multioutput), n_outputs))
    y_type = 'continuous' if n_outputs == 1 else 'continuous-multioutput'

    return y_type, y_true, y_pred, multioutput


from sklearn.metrics import mean_squared_error

from sklearn.model_selection import TimeSeriesSplit
from matplotlib import pyplot
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import statsmodels.api as sm
import itertools
import warnings


plt.style.use('Solarize_Light2')


#change csv file to appropriate Yahoo Finance CSV
df = pd.read_csv('UNVR-1Y.csv')

#Load data, convert date format
#Clean dividend row if necessary
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
print(df.index)
df.info()

#Dropping unused columns
df = df.drop(columns = ['Open','High','Low','Adj Close','Volume'])
print(df.head)

#Plotting
df.plot(figsize=(12,3));
plt.title('UNVR Price');

df['price_z'] = (df - df.rolling(window=10).mean()) / df.rolling(window=10).std()
df['price_zp'] = df['price_z'] - df['price_z'].shift(1)

fig, ax = plt.subplots(3,figsize=(12, 9))
ax[0].plot(df.index, df['Close'], label='raw data')
ax[0].plot(df['Close'].rolling(window=10).mean(), label="rolling mean");
ax[0].plot(df['Close'].rolling(window=10).std(), label="rolling std (x10)");
ax[0].legend()

ax[1].plot(df.index, df['price_z'], label="de-trended data")
ax[1].plot(df['price_z'].rolling(window=10).mean(), label="rolling mean");
ax[1].plot(df['price_z'].rolling(window=10).std(), label="rolling std (x10)");
ax[1].legend()

ax[2].plot(df.index, df['price_zp'], label="1 lag differenced de-trended data")
ax[2].plot(df['price_zp'].rolling(window=10).mean(), label="rolling mean");
ax[2].plot(df['price_zp'].rolling(window=10).std(), label="rolling std (x10)");
ax[2].legend()

plt.tight_layout()
fig.autofmt_xdate()


#ADF Test

print(" > Is the data stationary ?")
dftest = adfuller(df['Close'], autolag='AIC')
print("Test statistic = {:.3f}".format(dftest[0]))
print("P-value = {:.3f}".format(dftest[1]))
print("Critical values :")
for k, v in dftest[4].items():
    print("\t{}: {} - The data is {} stationary with {}% confidence".format(k, v, "not" if v<dftest[0] else "", 100-int(k[:-1])))
    
print("\n > Is the de-trended data stationary ?")
dftest = adfuller(df['price_z'].dropna(), autolag='AIC')
print("Test statistic = {:.3f}".format(dftest[0]))
print("P-value = {:.3f}".format(dftest[1]))
print("Critical values :")
for k, v in dftest[4].items():
    print("\t{}: {} - The data is {} stationary with {}% confidence".format(k, v, "not" if v<dftest[0] else "", 100-int(k[:-1])))
    
print("\n > Is the 12-lag differenced de-trended data stationary ?")
dftest = adfuller(df['price_zp'].dropna(), autolag='AIC')
print("Test statistic = {:.3f}".format(dftest[0]))
print("P-value = {:.3f}".format(dftest[1]))
print("Critical values :")
for k, v in dftest[4].items():
    print("\t{}: {} - The data is {} stationary with {}% confidence".format(k, v, "not" if v<dftest[0] else "", 100-int(k[:-1])))
    
#Price decomposition


decomp = seasonal_decompose(df['Close'], period=12)
decomp.plot()

# ACF/partial ACF eyeballing

fig, ax = plt.subplots(2, figsize=(12,6))
ax[0] = plot_acf(df['Close'].dropna(), ax=ax[0], lags=20)
ax[1] = plot_pacf(df['Close'].dropna(), ax=ax[1], lags=20)


# ARIMA Forecast

model = ARIMA(df['Close'].dropna(), order=(0, 0, 0))
mod_000 = model.fit()
print(mod_000.summary())

model = ARIMA(df['Close'].dropna(), order=(1, 0, 0))
mod_100 = model.fit()
print(mod_100.summary())

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
ax[0].plot(mod_000.resid.values, alpha=0.7, label='variance={:.3f}'.format(np.std(mod_000.resid.values)));
ax[0].hlines(0, xmin=0, xmax=350, color='r');
ax[0].set_title("ARIMA (0, 0, 0)");
ax[0].legend();

ax[1].plot(mod_100.resid.values, alpha=0.7, label='variance={:.3f}'.format(np.std(mod_100.resid.values)));
ax[1].hlines(0, xmin=0, xmax=350, color='r');
ax[1].set_title("ARIMA (1, 0, 0)");
ax[1].legend();

pred = mod_100.get_prediction(dynamic=False)
y_pred = pred.predicted_mean
sse = np.sqrt(np.mean(np.square(y_pred - df['Close'])))
mape = np.mean(np.abs((y_pred - df['Close'])/df['Close']))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Close'], label='historical');
ax.plot(y_pred, linestyle='--', color='#ff7823', label="ARIMA 100, RMSE={:0.2f}, MAPE={:0.2f}%, AIC={:0.2f}".format(sse, mape*100, mod_100.aic));
ax.legend();
ax.set_title("ARIMA");
plt.close()

#Splitting train-test

df2 = df.reset_index()

train_shares, test_shares = df2['Close'][0:-30], df2['Close'][-30:]

model_ev = ARIMA(train_shares.dropna(), order=(1, 0, 0))
mod_100_ev = model_ev.fit()
pred_ev = mod_100_ev.forecast(30)
ax = df2['Close'].plot(label='historical')
pred_ev.plot(linestyle='--', color='#ff7823', ax=ax, label='forecast', alpha=.7, figsize=(14, 7))

ax.set_xlabel('date')
ax.set_ylabel('price')
plt.legend()
plt.show()

# Quality matrices

rmse = lambda act, pred: np.sqrt(mean_squared_error(act, pred))

print('RMSE: {:0.2f}'.format(rmse(test_shares, pred_ev)))
print('MAPE: {:0.2f}%'.format(mean_absolute_percentage_error(test_shares, pred_ev)*100))

# ARIMA multiple train/test splits

X = df['Close']
splits = TimeSeriesSplit(n_splits=3)
pyplot.figure(1)
index = 1
for train_index, test_index in splits.split(X):
    train_shares_mul = X[train_index]
    test_shares_mul = X[test_index]
    print('Observations: %d' % (len(train_shares_mul) + len(test_shares_mul)))
    print('Training Observations: %d' % (len(train_shares_mul)))
    print('Testing Observations: %d' % (len(test_shares_mul)))
    pyplot.subplot(310 + index)
    pyplot.plot(train_shares_mul)
    pyplot.plot(test_shares_mul)
    index += 1
pyplot.show()


df2 = df.reset_index()
X2 = df2['Close']
RMSE = []
MAPE = []

pyplot.figure(1)
index = 1
for train_index, test_index in splits.split(X2):
    train_shares_mul = X2[train_index]
    test_shares_mul = X2[test_index]
    model_mul = ARIMA(train_shares_mul.dropna(), order=(1, 0, 0))
    mod_100_mul = model_mul.fit()
    pred_shares_mul = mod_100_mul.forecast(len(test_shares_mul))
    
    RMSE.append(rmse(test_shares_mul, pred_shares_mul))
    MAPE.append(mean_absolute_percentage_error(test_shares_mul, pred_shares_mul))
    
    print('RMSE: {:0.2f}'.format(RMSE[index-1]))
    print('MAPE: {:0.2f}%'.format(MAPE[index-1]*100))
    pyplot.subplot(310 + index)
    pyplot.plot(train_shares_mul, label='train')
    pyplot.plot(pred_shares_mul, label='pred')
    pyplot.plot(test_shares_mul, label='test')
    pyplot.legend()
    index += 1
    
RMSE = np.mean(RMSE)
MAPE = np.mean(MAPE)
print('RMSE_mean: {:0.2f}'.format(RMSE))
print('MAPE_mean: {:0.2f}%'.format(MAPE*100))
plt.close()

#ARIMA forecasting

model_fc = ARIMA(df2['Close'].dropna(), order=(1, 0, 0))
result = model_fc.fit()

fc = result.forecast(30)
for i in range(30):
    print("Projection of day "+str(i+1)+": %.0f" % (fc[i:i+1]))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df2['Close'], label='historical');
ax.plot(fc, linestyle='--', color='#ff7823', label="forecast");
ax.legend();
ax.set_title("ARIMA Forecasting");

# SARIMA Forecasting
# If you assume the price is seasonal

#Should be the same with the csv file above
saham = pd.read_csv('UNVR-1Y.csv')
saham = saham.set_index('Date')

saham = saham.drop(columns = ['Open','High','Low','Adj Close','Volume'])

saham.head()

harga = saham['Close']

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
# This one assumes 30-period seasonality
seasonal_pdq = [(x[0], x[1], x[2], 30) for x in list(itertools.product(p, d, q))]

warnings.filterwarnings("ignore") # specify to ignore warning messages

# Use dropna() since this would be out of sample

best_result = [0, 0, 10000000]
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(harga.dropna(),
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()
            
            if results.aic < best_result[2]:
                best_result = [param, param_seasonal, results.aic]
        except:
            continue

mod = sm.tsa.statespace.SARIMAX(harga.dropna(),
                                order=best_result[0],
                                seasonal_order=best_result[1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

best_result

pred_sar = results.get_prediction(dynamic=False)
y_pred_sar = pred_sar.predicted_mean
sse = np.sqrt(np.mean(np.square(y_pred_sar - saham['Close'])))
mape = np.mean(np.abs((y_pred_sar - saham['Close'])/saham['Close']))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(saham['Close'], label='historical');
ax.plot(y_pred_sar, linestyle='--', color='#ff7823', label="SARIMA, RMSE={:0.2f}, MAPE={:0.2f}%, AIC={:0.2f}".format(sse, mape*100, results.aic));
ax.legend();
ax.set_title("SARIMA");

# Split last 30 data
train_saham, test_saham = saham['Close'][0:-30], saham['Close'][-30:]

mod_ev = sm.tsa.statespace.SARIMAX(train_saham,
                                order=best_result[0],
                                seasonal_order=best_result[1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results_ev = mod_ev.fit()

# Change the start and end here
# Next step is finding start-end workaround for out of sample/skipped dates

pred_sar_ev = results_ev.get_prediction(start=212, end=241, dynamic=False)
pred_sar_ci = pred_sar_ev.conf_int()
ax = saham['Close'].plot(label='historical')
pred_sar_ev.predicted_mean.plot(linestyle='--', color='#ff7823', ax=ax, label='forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_sar_ci.index,
                pred_sar_ci.iloc[:, 0],
                pred_sar_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Price')
plt.legend()
plt.show()


forecast_ev = pred_sar_ev.predicted_mean
for i in range(30):
     print("Projected days + "+str(i+1)+": %.2f" % (forecast_ev[i:i+1]))

print('RMSE: {:0.2f}'.format(rmse(test_saham, forecast_ev)))
print('MAPE: {:0.2f}%'.format(mean_absolute_percentage_error(test_saham, forecast_ev)*100))
plt.close()

# SARIMA multiple train-test splits
X = saham['Close']
splits = TimeSeriesSplit(n_splits=2)
pyplot.figure(1)
index = 1
for train_index, test_index in splits.split(X):
    train_saham_mul = X[train_index]
    test_saham_mul = X[test_index]
    print('Observations: %d' % (len(train_saham_mul) + len(test_saham_mul)))
    print('Training Observations: %d' % (len(train_saham_mul)))
    print('Testing Observations: %d' % (len(test_saham_mul)))
    pyplot.subplot(310 + index)
    pyplot.plot(train_saham_mul)
    pyplot.plot(test_saham_mul)
    index += 1
pyplot.show()


RMSE = []
MAPE = []
pyplot.figure(1)
index = 1
for train_index, test_index in splits.split(X):
    train_saham_mul = X[train_index]
    test_saham_mul = X[test_index]
    model_saham_mul = sm.tsa.statespace.SARIMAX(train_saham_mul,
                                order=best_result[0],
                                seasonal_order=best_result[1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    results_saham_mul = model_saham_mul.fit()
    pred_saham_mul = results_saham_mul.forecast(len(test_saham_mul))
    
    RMSE.append(rmse(test_saham_mul, pred_saham_mul))
    MAPE.append(mean_absolute_percentage_error(test_saham_mul, pred_saham_mul))
    print('RMSE: {:0.2f}'.format(RMSE[index-1]))
    print('MAPE: {:0.2f}%'.format(MAPE[index-1]*100))
    
    pyplot.subplot(310 + index)
    pyplot.plot(train_saham_mul, label='train')
    pyplot.plot(pred_saham_mul, label='pred')
    pyplot.plot(test_saham_mul, label='test')
    pyplot.legend()
    index += 1

RMSE = np.mean(RMSE)
MAPE = np.mean(MAPE)
print('RMSE_mean: {:0.2f}'.format(RMSE))
print('MAPE_mean: {:0.2f}%'.format(MAPE*100))

RMSE = np.mean(RMSE)
MAPE = np.mean(MAPE)
print('RMSE_mean: {:0.2f}'.format(RMSE))
print('MAPE_mean: {:0.2f}%'.format(MAPE*100))
plt.close()

# Exponential Smoothing

# Generating forecasting options
tr = ['add', 'mul']
ss = ['add', 'mul']
dp = [True, False]
combs = {}
aics = []

# Iterating forecasting options
# Change seasonal period appropriately
for i in tr:
    for j in ss:
        for k in dp:
            model = ExponentialSmoothing(harga.dropna(), trend=i, seasonal=j, seasonal_periods=30, damped=k)
            model = model.fit()
            combs.update({model.aic : [i, j, k]})
            aics.append(model.aic)

# Forecasting using best fit model
# Akaike criterion could be replaced
best_aic = min(aics)
model = ExponentialSmoothing(harga.dropna(), trend=combs[best_aic][0], seasonal=combs[best_aic][1], seasonal_periods=30, damped=combs[best_aic][2])
results_es = model.fit()
y_pred_es = results_es.predict(0,len(saham['Close'])-1)
sse = np.sqrt(np.mean(np.square(y_pred_es - saham['Close'])))
mape = np.mean(np.abs((y_pred_es - saham['Close'])/saham['Close']))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(saham['Close'], label='historical');
ax.plot(y_pred_es, linestyle='--', color='#ff7823', label="EXP. SMOOTHING, RMSE={:0.2f}, MAPE={:0.2f}%, AIC={:0.2f}".format(sse, mape*100, results_es.aic));
ax.legend();
ax.set_title("EXP. SMOOTHING");

tabel = []
for i in tr:
    for j in ss:
        for k in dp:
            model = ExponentialSmoothing(harga.dropna(), trend=i, seasonal=j, seasonal_periods=30, damped=k)
            model = model.fit()
            pred = model.predict(0,len(harga)-1)
            sse = np.sqrt(np.mean(np.square(pred - harga)))
            mape = np.mean(np.abs((pred - harga)/harga))
            lst = [i, j, k, model.aic, "{:0.2f}".format(sse), "{:0.2f}%".format(mape*100)]
            tabel.append(lst)

ev_table = pd.DataFrame(tabel, columns=['trend','seasonality','damped','AIC','RMSE','MAPE (%)'])
ev_table

model_ev = ExponentialSmoothing(train_saham, trend=combs[best_aic][0], seasonal=combs[best_aic][1], seasonal_periods=30, damped=combs[best_aic][2])
results_es_ev = model_ev.fit()

forecast_es_ev = results_es_ev.forecast(30)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(saham['Close'], label='historical');
ax.plot(forecast_es_ev, linestyle='--', color='#ff7823', label="forecast");
ax.legend();
ax.set_title("Exp. Smoothing Forecasting");

for i in range(12):
     print("Projected days + "+str(i+1)+": %.2f" % (forecast_es_ev[i:i+1]))

print('RMSE: {:0.2f}'.format(rmse(test_saham, forecast_es_ev)))
print('MAPE: {:0.2f}%'.format(mean_absolute_percentage_error(test_saham, forecast_es_ev)*100))
plt.close()

# Exp. Smoothing - multiple train-test splits
RMSE = []
MAPE = []
pyplot.figure(1)
index = 1
for train_index, test_index in splits.split(X):
    train_es_mul = X[train_index]
    test_es_mul = X[test_index]
    model_es_mul = ExponentialSmoothing(train_es_mul, trend=combs[best_aic][0], seasonal=combs[best_aic][1], seasonal_periods=30, damped=combs[best_aic][2])
    results_es_mul = model_es_mul.fit()
    pred_es_mul = results_es_mul.forecast(len(test_es_mul))
    
    RMSE.append(rmse(test_es_mul, pred_es_mul))
    MAPE.append(mean_absolute_percentage_error(test_es_mul, pred_es_mul))
    print('RMSE: {:0.2f}'.format(RMSE[index-1]))
    print('MAPE: {:0.2f}%'.format(MAPE[index-1]*100))
    
    pyplot.subplot(310 + index)
    pyplot.plot(train_es_mul, label='train')
    pyplot.plot(pred_es_mul, label='pred')
    pyplot.plot(test_es_mul, label='test')
    pyplot.legend()
    index += 1

RMSE = np.mean(RMSE)
MAPE = np.mean(MAPE)
print('RMSE_mean: {:0.2f}'.format(RMSE))
print('MAPE_mean: {:0.2f}%'.format(MAPE*100))

# Exp. Smooting forecast

fc_es = results_es.forecast(5)
for i in range(5):
    print("Projection of days"+str(i+8)+": %.2f" % (fc_es[i:i+1]))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(saham['Close'], label='historical');
ax.plot(fc_es, linestyle='--', color='#ff7823', label="forecast");
ax.legend();
ax.set_title("Exp. Smoothing Forecasting");