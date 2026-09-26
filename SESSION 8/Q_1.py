import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.array([
    [4, 64],
    [6, 128],
    [8, 256],
    [12, 512]
])

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

print("Original Feature Matrix:")
print(X)

print("\nPolynomial Feature Matrix:")
print(X_poly)

print("\nFeature Names:")
print(poly.get_feature_names_out(["RAM", "Storage"]))