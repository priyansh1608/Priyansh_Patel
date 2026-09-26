import pandas as pd

df = pd.read_csv("SESSION 21\hmedabad_temperature.csv")

df["date"] = pd.to_datetime(df["date"])

df["temperature_lag_3"] = df["temperature"].shift(3)
df["rolling_average_7"] = df["temperature"].rolling(7).mean()

print(df)