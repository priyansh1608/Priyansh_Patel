from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

data = fetch_california_housing()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Lasso(alpha=0.05)
model.fit(X_train, y_train)

kept_features = []
dropped_features = []

for feature, coefficient in zip(data.feature_names, model.coef_):
    if coefficient != 0:
        kept_features.append(feature)
    else:
        dropped_features.append(feature)

print("Kept Features:")
for feature in kept_features:
    print(feature)

print("\nDropped Features:")
for feature in dropped_features:
    print(feature)

print("\nFeature Coefficients:")

for feature, coefficient in zip(data.feature_names, model.coef_):
    print(feature, ":", coefficient)