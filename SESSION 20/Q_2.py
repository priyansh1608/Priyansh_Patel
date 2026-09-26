import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("SESSION 20\emperature.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")

result = seasonal_decompose(
    df["temperature"],
    model="additive",
    period=7
)

result.plot()
plt.tight_layout()
plt.show()