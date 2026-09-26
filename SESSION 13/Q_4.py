from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

decision_tree = DecisionTreeClassifier(
    random_state=42
)

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

decision_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)

tree_prediction = decision_tree.predict(X_test)
forest_prediction = random_forest.predict(X_test)

tree_accuracy = accuracy_score(y_test, tree_prediction)
forest_accuracy = accuracy_score(y_test, forest_prediction)

print("Decision Tree Accuracy:", tree_accuracy)
print("Random Forest Accuracy:", forest_accuracy)

print("\nDecision Tree Accuracy Percentage:", tree_accuracy * 100, "%")
print("Random Forest Accuracy Percentage:", forest_accuracy * 100, "%")