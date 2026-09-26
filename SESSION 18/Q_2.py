from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

X = iris.data
y = iris.target

model = DecisionTreeClassifier(random_state=42)

scores = cross_val_score(model, X, y, cv=10)

print("Accuracy for Each Fold:")

for i, score in enumerate(scores, start=1):
    print("Fold", i, ":", score)

print("Average Accuracy:", scores.mean())
print("Average Accuracy Percentage:", scores.mean() * 100, "%")