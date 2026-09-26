from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data = fetch_california_housing()

X = data.data
y = data.target

print("Feature Names:")
print(data.feature_names)

print("\nX Shape:")
print(X.shape)

print("\ny Shape:")
print(y.shape)

print("\nFirst 5 rows of X:")
print(X[:5])

print("\nFirst 5 target values:")
print(y[:5])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)