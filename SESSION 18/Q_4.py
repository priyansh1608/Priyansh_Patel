from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

wine = load_wine()

X = wine.data
y = wine.target

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

cv_values = [3, 5, 10]

for cv in cv_values:
    scores = cross_val_score(model, X, y, cv=cv)

    print("CV =", cv)
    print("Scores:", scores)
    print("Mean Accuracy:", scores.mean())
    print("Standard Deviation:", scores.std())
    print()