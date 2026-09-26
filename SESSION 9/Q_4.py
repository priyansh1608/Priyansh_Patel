from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet

data = fetch_california_housing()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lasso = Lasso(alpha=0.01)
ridge = Ridge(alpha=1.0)
elastic = ElasticNet(alpha=0.01, l1_ratio=0.5)

lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)
elastic.fit(X_train, y_train)

print("Feature\t\tLasso\t\tRidge\t\tElasticNet")

for feature, lasso_coef, ridge_coef, elastic_coef in zip(
    data.feature_names,
    lasso.coef_,
    ridge.coef_,
    elastic.coef_
):
    print(
        f"{feature}\t\t{lasso_coef:.6f}\t{ridge_coef:.6f}\t{elastic_coef:.6f}"
    )

print("\nElasticNet with different l1_ratio values:")

for ratio in [0.1, 0.5, 0.9]:
    model = ElasticNet(alpha=0.01, l1_ratio=ratio)
    model.fit(X_train, y_train)

    print("\nl1_ratio =", ratio)

    for feature, coefficient in zip(data.feature_names, model.coef_):
        print(feature, ":", round(coefficient, 6))