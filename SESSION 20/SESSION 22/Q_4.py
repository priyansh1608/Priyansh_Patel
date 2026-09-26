import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

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

df = df.dropna(subset=["rating", "cost", "cuisine", "location"])

X = df[["cuisine", "location", "cost"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), ["cuisine", "location"]),
    ("numeric", StandardScaler(), ["cost"])
])

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

logistic_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

logistic_probability = logistic_model.predict_proba(X_test)[:, 1]
random_forest_probability = random_forest_model.predict_proba(X_test)[:, 1]

logistic_auc = roc_auc_score(y_test, logistic_probability)
random_forest_auc = roc_auc_score(y_test, random_forest_probability)

print("Logistic Regression ROC-AUC:", logistic_auc)
print("Random Forest ROC-AUC:", random_forest_auc)

if logistic_auc > random_forest_auc:
    print("Logistic Regression performed better.")
else:
    print("Random Forest performed better.")