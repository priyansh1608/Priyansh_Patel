from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

X = iris.data
y = iris.target

model = DecisionTreeClassifier(
    random_state=42
)

model.fit(X, y)

importance = model.feature_importances_

features = list(zip(iris.feature_names, importance))

features.sort(
    key=lambda x: x[1],
    reverse=True
)

print("Feature Importance:")
print()

for feature, score in features:
    print(feature, ":", score)