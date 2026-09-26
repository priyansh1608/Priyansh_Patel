import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("SESSION 21\hmedabad_temperature.csv")

df["date"] = pd.to_datetime(df["date"])

df["temperature_lag_3"] = df["temperature"].shift(3)
df["rolling_average_7"] = df["temperature"].rolling(7).mean()

df = df.dropna()

X = df[["temperature_lag_3", "rolling_average_7"]]
y = df["temperature"]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

last_lag = df["temperature_lag_3"].iloc[-1]
last_rolling = df["rolling_average_7"].iloc[-1]

prediction = model.predict([[last_lag, last_rolling]])

next_date = df["date"].iloc[-1] + pd.Timedelta(days=1)

print("Next Available Date:", next_date.date())
print("Predicted Temperature:", prediction[0])