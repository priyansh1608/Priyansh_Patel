import pandas as pd

df = pd.read_csv("SESSION 21\kart_sales.csv")

df["order_date"] = pd.to_datetime(df["order_date"])

df["day_of_week"] = df["order_date"].dt.day_name()
df["month"] = df["order_date"].dt.month
df["is_weekend"] = df["order_date"].dt.dayofweek >= 5

print(df)