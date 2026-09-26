from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

gini_model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

entropy_model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

gini_model.fit(X_train, y_train)
entropy_model.fit(X_train, y_train)

gini_prediction = gini_model.predict(X_test)
entropy_prediction = entropy_model.predict(X_test)

gini_accuracy = accuracy_score(y_test, gini_prediction)
entropy_accuracy = accuracy_score(y_test, entropy_prediction)

print("Gini Accuracy:", gini_accuracy)
print("Entropy Accuracy:", entropy_accuracy)