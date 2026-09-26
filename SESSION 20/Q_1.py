import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("SESSION 20\emperature.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")

plt.plot(df.index, df["temperature"])
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.title("Daily Temperature")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()