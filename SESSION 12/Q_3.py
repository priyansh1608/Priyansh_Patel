from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data
y = iris.target

model = DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)

model.fit(X, y)

plt.figure(figsize=(12, 8))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)

plt.title("Decision Tree with Maximum Depth 2")
plt.show()