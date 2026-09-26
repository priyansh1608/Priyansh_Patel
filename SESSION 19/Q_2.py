import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

data = load_breast_cancer()

X = data.data
y = data.target

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

param_distributions = {
    'svm__C': [0.01, 0.1, 1, 10, 100],
    'svm__gamma': ['scale', 0.001, 0.01, 0.1, 1]
}

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=10,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X, y)

print("Best Parameters:", random_search.best_params_)
print("Best Accuracy:", random_search.best_score_)