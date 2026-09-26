import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.DataFrame({
    "Runs": [450, 320, 280, 510, 150, 120, 390, 275, 480, 100, 210, 350, 500, 180, 260, 420, 140, 300, 470, 220],
    "StrikeRate": [145, 132, 128, 150, 110, 105, 140, 125, 148, 100, 115, 135, 152, 108, 120, 142, 102, 130, 149, 118],
    "Wickets": [2, 1, 0, 3, 8, 10, 1, 4, 2, 12, 7, 1, 3, 9, 5, 2, 11, 4, 1, 6],
    "PlayerType": [
        "Batsman", "Batsman", "Batsman", "Batsman", "Bowler",
        "Bowler", "Batsman", "AllRounder", "Batsman", "Bowler",
        "Bowler", "Batsman", "Batsman", "Bowler", "AllRounder",
        "Batsman", "Bowler", "AllRounder", "Batsman", "Bowler"
    ]
})

X = data[["Runs", "StrikeRate", "Wickets"]]
y = data["PlayerType"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

poly_model = SVC(kernel="poly", degree=3)
rbf_model = SVC(kernel="rbf")

poly_model.fit(X_train, y_train)
rbf_model.fit(X_train, y_train)

poly_pred = poly_model.predict(X_test)
rbf_pred = rbf_model.predict(X_test)

poly_accuracy = accuracy_score(y_test, poly_pred)
rbf_accuracy = accuracy_score(y_test, rbf_pred)

print("Polynomial Kernel Accuracy:", poly_accuracy)
print("Polynomial Kernel Percentage:", poly_accuracy * 100, "%")

print("RBF Kernel Accuracy:", rbf_accuracy)
print("RBF Kernel Percentage:", rbf_accuracy * 100, "%")

if poly_accuracy > rbf_accuracy:
    print("Polynomial kernel performed better on this dataset.")
elif rbf_accuracy > poly_accuracy:
    print("RBF kernel performed better on this dataset.")
else:
    print("Both kernels performed equally on this dataset.")