import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("SESSION 20\emperature.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")

df["moving_average"] = df["temperature"].rolling(window=7).mean()

plt.plot(df.index, df["temperature"], label="Original")
plt.plot(df.index, df["moving_average"], label="7-Day Moving Average")

plt.xlabel("Date")
plt.ylabel("Temperature")
plt.title("Original vs Moving Average")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()