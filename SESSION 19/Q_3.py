import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

data = load_breast_cancer()

X = data.data
y = data.target

class_0 = np.where(y == 0)[0]
class_1 = np.where(y == 1)[0]

np.random.seed(42)

selected_class_1 = np.random.choice(class_1, size=100, replace=False)

selected_indices = np.concatenate([class_0, selected_class_1])

X = X[selected_indices]
y = y[selected_indices]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42))
])

param_grid = {
    'rf__n_estimators': [50, 100],
    'rf__max_depth': [5, 10, None]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_test)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross Validation F1 Score:", grid_search.best_score_)
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))