from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

parameters = [
    (50, 0.1),
    (100, 0.1),
    (150, 0.1),
    (50, 0.05),
    (100, 0.05),
    (150, 0.05),
    (50, 0.2),
    (100, 0.2),
    (150, 0.2)
]

best_accuracy = 0
best_n_estimators = 0
best_learning_rate = 0

for n_estimators, learning_rate in parameters:
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(
        "n_estimators:", n_estimators,
        "learning_rate:", learning_rate,
        "Accuracy:", accuracy
    )

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n_estimators = n_estimators
        best_learning_rate = learning_rate

print("\nBest Parameters:")
print("n_estimators:", best_n_estimators)
print("learning_rate:", best_learning_rate)
print("Best Accuracy:", best_accuracy)
print("Best Accuracy Percentage:", best_accuracy * 100, "%")