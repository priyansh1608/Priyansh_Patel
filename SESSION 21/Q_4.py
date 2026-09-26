import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("SESSION 21\hmedabad_temperature.csv")

df["date"] = pd.to_datetime(df["date"])

df["temperature_lag_3"] = df["temperature"].shift(3)
df["rolling_average_7"] = df["temperature"].rolling(7).mean()

df = df.dropna()

X = df[["temperature_lag_3", "rolling_average_7"]]
y = df["temperature"]

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

def mape(actual, predicted):
    return np.mean(
        np.abs((actual - predicted) / actual)
    ) * 100

mape_value = mape(y_test.values, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape_value, "%")

print("RMSE is more sensitive to large errors because it squares the errors.")