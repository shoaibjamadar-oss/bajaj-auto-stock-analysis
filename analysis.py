import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import timedelta

# loading the dataset
df = pd.read_csv("BAJAJ_AUTO.csv")
df.columns = df.columns.str.strip()

# keeping only date and close price
df = df[['DATE', 'CLOSE']]
df.columns = ['Date', 'Close']

# removing commas from numbers like 9,800 -> 9800
df['Close'] = df['Close'].astype(str).str.replace(',', '').str.strip()
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# removing missing values
df = df.dropna()
df = df.sort_values('Date').reset_index(drop=True)

print("Data loaded successfully")
print("Total rows:", len(df))
print(df.head())

# plotting closing price
plt.figure(figsize=(12,5))
plt.plot(df['Date'], df['Close'], color='blue')
plt.title('BAJAJ-AUTO Closing Price')
plt.xlabel('Date')
plt.ylabel('Price (Rs)')
plt.xticks(rotation=30)
plt.grid(True)
plt.tight_layout()
plt.savefig('01_closing_price.png')
plt.show()

# ADF test to check if data is stationary
close = df['Close'].values

adf1 = adfuller(close)
print("\nADF Test on original data:")
print("Test Statistic =", round(adf1[0], 4))
print("p-value =", round(adf1[1], 4))
if adf1[1] < 0.05:
    print("Data is Stationary, d = 0")
    d = 0
else:
    print("Data is Non-Stationary, d = 1")
    d = 1

adf2 = adfuller(np.diff(close))
print("\nADF Test after differencing:")
print("p-value =", round(adf2[1], 4))
if adf2[1] < 0.05:
    print("Data is now Stationary")

# ACF and PACF plots to find p and q
if d == 1:
    series = np.diff(close)
else:
    series = close

fig, axes = plt.subplots(1, 2, figsize=(12,4))
plot_acf(series, lags=20, ax=axes[0])
plot_pacf(series, lags=20, ax=axes[1], method='ywm')
plt.suptitle('ACF and PACF Plots')
plt.tight_layout()
plt.savefig('02_acf_pacf.png')
plt.show()

# from ACF and PACF we selected p=2 and q=2
p = 2
q = 2
print(f"\nARIMA order = ({p},{d},{q})")

# splitting data into train and test
split = int(len(close) * 0.8)
train = close[:split]
test = close[split:]
train_dates = df['Date'].values[:split]
test_dates = df['Date'].values[split:]

print("Train size:", len(train))
print("Test size:", len(test))

# fitting ARIMA model
model = ARIMA(train, order=(p, d, q))
result = model.fit()
print(result.summary())

# predicting on test data
pred = result.forecast(steps=len(test))

# calculating errors
rmse = np.sqrt(mean_squared_error(test, pred))
mae = mean_absolute_error(test, pred)
mape = np.mean(np.abs((test - pred) / test)) * 100

print("\nModel Performance:")
print("RMSE =", round(rmse, 2))
print("MAE =", round(mae, 2))
print("MAPE =", round(mape, 2), "%")

# plotting actual vs predicted
plt.figure(figsize=(12,5))
plt.plot(train_dates, train, color='blue', label='Train Data')
plt.plot(test_dates, test, color='green', label='Actual')
plt.plot(test_dates, pred, color='red', linestyle='--', label='Predicted')
plt.title('ARIMA Model - Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Price (Rs)')
plt.legend()
plt.xticks(rotation=30)
plt.grid(True)
plt.tight_layout()
plt.savefig('03_arima_eval.png')
plt.show()

# forecasting next 30 days
full_model = ARIMA(close, order=(p, d, q))
full_result = full_model.fit()

forecast = full_result.get_forecast(steps=30)
forecast_values = forecast.predicted_mean
conf = forecast.conf_int(alpha=0.05)

# creating future dates
last_date = df['Date'].max()
future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=30)

# saving forecast to csv
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecasted Price': forecast_values,
    'Lower Bound': conf[:, 0],
    'Upper Bound': conf[:, 1]
})
forecast_df.to_csv('forecast_30days.csv', index=False)
print("\nForecast for next 30 days:")
print(forecast_df[['Date', 'Forecasted Price']].to_string(index=False))

# plotting forecast with last 90 days
plt.figure(figsize=(12,5))
plt.plot(df['Date'].tail(90), df['Close'].tail(90), color='blue', label='Historical Data')
plt.plot(future_dates, forecast_values, color='red', linestyle='--', label='Forecast')
plt.fill_between(future_dates, conf[:, 0], conf[:, 1], alpha=0.2, color='red', label='Confidence Interval')
plt.title('BAJAJ-AUTO - Next 30 Days Forecast')
plt.xlabel('Date')
plt.ylabel('Price (Rs)')
plt.legend()
plt.xticks(rotation=35)
plt.grid(True)
plt.tight_layout()
plt.savefig('04_forecast.png')
plt.show()

# interpretation
last_price = close[-1]
forecast_day30 = forecast_values[-1]
change = ((forecast_day30 - last_price) / last_price) * 100

print("\nInterpretation:")
print("Last closing price = Rs", round(last_price, 2))
print("Forecasted price after 30 days = Rs", round(forecast_day30, 2))
print("Expected change =", round(change, 2), "%")

if change > 1:
    print("The model shows an UPWARD trend")
elif change < -1:
    print("The model shows a DOWNWARD trend")
else:
    print("The model shows a STABLE trend")

print("\nDone! All 4 graphs and forecast CSV saved in your folder.")
