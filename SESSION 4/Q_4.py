import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE

df = pd.read_csv("SESSION 4\zomato_data.csv")

features = [
    "online_order",
    "book_table",
    "votes",
    "rating",
    "price_range"
]

X = df[features]
y = df["average_cost_for_two"]

model = DecisionTreeRegressor(random_state=42)

rfe = RFE(
    estimator=model,
    n_features_to_select=3
)

rfe.fit(X, y)

selected_features = X.columns[rfe.support_]

print("Selected 3 features:")
for feature in selected_features:
    print(feature)

print("\nRFE Ranking:")
for feature, ranking in zip(X.columns, rfe.ranking_):
    print(feature, "=", ranking)