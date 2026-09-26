import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

data = {
    "runs": [45, 120, 15, 80, 200, 35, 150, 25, 95, 180, 10, 70],
    "wickets": [2, 8, 1, 5, 9, 2, 7, 1, 4, 8, 0, 5],
    "strike_rate": [110, 145, 85, 135, 160, 105, 150, 90, 140, 155, 75, 130],
    "win": [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[["runs", "wickets", "strike_rate"]]
y = df["win"]

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X, y)

plt.figure(figsize=(14, 9))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Loss", "Win"],
    filled=True
)

plt.title("IPL Match Result Decision Tree")
plt.show()

print("First Three Splits:")

tree = model.tree_

for i in range(3):
    feature_index = tree.feature[i]
    threshold = tree.threshold[i]

    if feature_index >= 0:
        print(
            "Split", i + 1,
            ":", X.columns[feature_index],
            "<=",
            round(threshold, 2)
        )