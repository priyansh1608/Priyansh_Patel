from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model_before = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model_before.fit(X_train, y_train)

y_pred_before = model_before.predict(X_test)

accuracy_before = accuracy_score(y_test, y_pred_before)
f1_before = f1_score(y_test, y_pred_before)

model_after = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

model_after.fit(X_train, y_train)

y_pred_after = model_after.predict(X_test)

accuracy_after = accuracy_score(y_test, y_pred_after)
f1_after = f1_score(y_test, y_pred_after)

print("Before class_weight='balanced'")
print("Accuracy:", accuracy_before)
print("F1 Score:", f1_before)

print()

print("After class_weight='balanced'")
print("Accuracy:", accuracy_after)
print("F1 Score:", f1_after)