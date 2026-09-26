from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

X = iris.data
y = iris.target

model = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(model, X, y, cv=5)

print("Cross Validation Scores:")
print(scores)

print("Average Accuracy:", scores.mean())
print("Average Accuracy Percentage:", scores.mean() * 100, "%")