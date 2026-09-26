from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge

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

lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)

print("Feature\t\tLasso\t\tRidge")

for feature, lasso_coef, ridge_coef in zip(
    data.feature_names, lasso.coef_, ridge.coef_
):
    print(
        f"{feature}\t\t{lasso_coef:.6f}\t{ridge_coef:.6f}"
    )

print("\nFeatures with zero Lasso coefficients:")

for feature, coefficient in zip(data.feature_names, lasso.coef_):
    if coefficient == 0:
        print(feature)