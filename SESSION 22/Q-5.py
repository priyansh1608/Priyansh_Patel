import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

df = pd.read_csv("SESSION 22\zomato.csv")

df = df.rename(columns={
    "rate": "rating",
    "approx_cost(for two)": "cost",
    "cuisines": "cuisine"
})

df["rating"] = pd.to_numeric(
    df["rating"].astype(str).str.replace("/5", "", regex=False),
    errors="coerce"
)

df["cost"] = pd.to_numeric(
    df["cost"].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)

df["target"] = (df["rating"] > 4.0).astype(int)

df = df.dropna(
    subset=["rating", "cost", "cuisine", "location"]
)

X = df[["cuisine", "location", "cost"]]
y = df["target"]

print("Class Distribution Before SMOTE:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), ["cuisine", "location"])
], remainder="passthrough")

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(random_state=42))
])

param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [5, 10, 20]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

probabilities = grid_search.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation ROC-AUC:")
print(grid_search.best_score_)

print("\nTest ROC-AUC:")
print(roc_auc)