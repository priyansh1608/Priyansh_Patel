import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("SESSION 20\emperature.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")

model = ARIMA(df["temperature"], order=(1, 1, 1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=7)

print("Next 7 Days Temperature Forecast:")
print(forecast)

future_dates = pd.date_range(
    start=df.index[-1] + pd.Timedelta(days=1),
    periods=7
)

plt.plot(df.index, df["temperature"], label="Original")
plt.plot(future_dates, forecast, label="Forecast")

plt.xlabel("Date")
plt.ylabel("Temperature")
plt.title("Temperature Forecast for Next 7 Days")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()