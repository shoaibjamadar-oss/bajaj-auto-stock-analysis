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

# ── LOAD DATA ──────────────────────────────────────────────────
df = pd.read_csv("bajaj_auto.csv")
df.columns = df.columns.str.strip()
df = df[['DATE', 'CLOSE']].rename(columns={'DATE': 'Date', 'CLOSE': 'Close'})

# Remove commas from numbers and convert
df['Close'] = df['Close'].astype(str).str.replace(',', '').str.strip()
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['Date']  = pd.to_datetime(df['Date'], dayfirst=True)
df = df.dropna()
df = df.sort_values('Date').reset_index(drop=True)

print("Rows:", len(df))
print(df.head())

# ── (i-c) CLOSING PRICE TREND ──────────────────────────────────
plt.figure(figsize=(12, 4))
plt.plot(df['Date'], df['Close'], color='blue')
plt.title('BAJAJ-AUTO Closing Price - Past 1 Year')
plt.xlabel('Date')
plt.ylabel('Price (Rs)')
plt.xticks(rotation=30)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('01_closing_price.png', dpi=150)
plt.show()
print("Saved: 01_closing_price.png")

# ── (ii-a) ADF TEST ────────────────────────────────────────────
close = df['Close'].values

r1 = adfuller(close)
print("\nADF Test - Original Series")
print("Test Statistic:", round(r1[0], 4))
print("p-value:", round(r1[1], 4))
print("Result:", "STATIONARY" if r1[1] < 0.05 else "NON-STATIONARY")

r2 = adfuller(np.diff(close))
print("\nADF Test - 1st Difference")
print("Test Statistic:", round(r2[0], 4))
print("p-value:", round(r2[1], 4))
print("Result:", "STATIONARY" if r2[1] < 0.05 else "NON-STATIONARY")

d = 0 if r1[1] < 0.05 else 1
print("\nChosen d =", d)

# ── (ii-b) ACF AND PACF ────────────────────────────────────────
series = np.diff(close) if d == 1 else close

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(series,  lags=20, ax=axes[0], title='ACF Plot')
plot_pacf(series, lags=20, ax=axes[1], title='PACF Plot', method='ywm')
plt.tight_layout()
plt.savefig('02_acf_pacf.png', dpi=150)
plt.show()
print("Saved: 02_acf_pacf.png")

p, q = 2, 2
print(f"ARIMA order chosen: ({p},{d},{q})")

# ── (ii-c) FIT ARIMA ───────────────────────────────────────────
split      = int(len(close) * 0.8)
train      = close[:split]
test       = close[split:]
train_dates = df['Date'].values[:split]
test_dates  = df['Date'].values[split:]

model  = ARIMA(train, order=(p, d, q))
fitted = model.fit()
pred   = fitted.forecast(steps=len(test))

rmse = np.sqrt(mean_squared_error(test, pred))
mae  = mean_absolute_error(test, pred)
mape = np.mean(np.abs((test - pred) / test)) * 100
print(f"\nRMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}%")

plt.figure(figsize=(12, 4))
plt.plot(train_dates, train, color='blue',  label='Train')
plt.plot(test_dates,  test,  color='green', label='Actual')
plt.plot(test_dates,  pred,  color='red', linestyle='--', label='Predicted')
plt.title(f'ARIMA({p},{d},{q}) - Train vs Test | MAPE={mape:.2f}%')
plt.xlabel('Date')
plt.ylabel('Price (Rs)')
plt.legend()
plt.xticks(rotation=30)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('03_arima_eval.png', dpi=150)
plt.show()
print("Saved: 03_arima_eval.png")

# ── (iii) 30-DAY FORECAST ──────────────────────────────────────
full    = ARIMA(close, order=(p, d, q)).fit()
fc      = full.get_forecast(steps=30)
fc_vals = fc.predicted_mean
fc_ci   = fc.conf_int(alpha=0.05)

future = pd.bdate_range(start=df['Date'].max() + timedelta(days=1), periods=30)

fc_df = pd.DataFrame({
    'Date':   future,
    'Forecast': fc_vals,
    'Lower':    fc_ci[:, 0],
    'Upper':    fc_ci[:, 1]
})
fc_df.to_csv('forecast_30days.csv', index=False)
print("\n30 Day Forecast:")
print(fc_df[['Date', 'Forecast']].to_string(index=False))
print("Saved: forecast_30days.csv")

plt.figure(figsize=(12, 5))
plt.plot(df['Date'].tail(90), df['Close'].tail(90), color='blue', label='Last 90 Days')
plt.plot(future, fc_vals, color='red', linestyle='--', marker='o', markersize=3, label='30-Day Forecast')
plt.fill_between(future, fc_ci[:, 0], fc_ci[:, 1], alpha=0.15, color='red', label='95% Confidence')
plt.axvline(x=df['Date'].max(), color='grey', linestyle=':')
plt.title('BAJAJ-AUTO - 30 Day Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price (Rs)')
plt.legend()
plt.xticks(rotation=35)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('04_forecast.png', dpi=150)
plt.show()
print("Saved: 04_forecast.png")

# ── (iv) INTERPRETATION ────────────────────────────────────────
change = ((fc_vals[-1] - close[-1]) / close[-1]) * 100
print(f"\nLast Price: Rs{close[-1]:.2f}")
print(f"Day 30 Forecast: Rs{fc_vals[-1]:.2f}")
print(f"Change: {change:+.2f}%")
print("Trend:", "UPWARD" if change > 1 else ("DOWNWARD" if change < -1 else "STABLE"))

