from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

categories = ['rec.sport.baseball', 'sci.med']

data = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)

X = data.data
y = data.target

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=2000)),
    ('rf', RandomForestClassifier(random_state=42))
])

param_grid = {
    'rf__max_depth': [5, 10, 20],
    'rf__n_estimators': [50, 100]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X, y)

print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)