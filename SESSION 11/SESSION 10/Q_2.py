import pandas as pd
from sklearn.linear_model import LogisticRegression

data = {
    "likes": [50, 120, 300, 80, 500, 1000, 60, 800, 150, 600],
    "has_caption": [1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    "viral": [0, 0, 1, 0, 1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[["likes", "has_caption"]]
y = df["viral"]

model = LogisticRegression()
model.fit(X, y)

print("Model Coefficients:")
print("Likes:", model.coef_[0][0])
print("Has Caption:", model.coef_[0][1])

print("\nIntercept:")
print(model.intercept_[0])