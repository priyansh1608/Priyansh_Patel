from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
import numpy as np

iris = load_iris()

X = iris.data
y = iris.target

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
    y_train = y[train_index]
    y_test = y[test_index]

    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)

    print("Fold", fold)
    print("Training class counts:", train_counts)
    print("Testing class counts:", test_counts)
    print()