from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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

n_estimators_values = [10, 50, 100]
max_depth_values = [2, 4, 6]

best_accuracy = 0
best_n_estimators = 0
best_max_depth = 0

for n_estimators in n_estimators_values:
    for max_depth in max_depth_values:

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        accuracy = accuracy_score(y_test, prediction)

        print(
            "n_estimators:",
            n_estimators,
            "max_depth:",
            max_depth,
            "accuracy:",
            accuracy
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_n_estimators = n_estimators
            best_max_depth = max_depth

print("\nBest Combination:")
print("n_estimators:", best_n_estimators)
print("max_depth:", best_max_depth)
print("Best Accuracy:", best_accuracy)