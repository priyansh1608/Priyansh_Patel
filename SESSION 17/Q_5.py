import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

data = pd.read_csv("creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

test_data = pd.DataFrame(X_test)
test_data["target"] = y_test.to_numpy()

positive = test_data[test_data["target"] == 1]

negative = test_data[test_data["target"] == 0]

desired_positive = max(1, int(len(test_data) * 0.02 / 0.98))

if len(positive) >= desired_positive:
    positive = positive.sample(
        n=desired_positive,
        random_state=42
    )

negative_count = int(len(positive) * 98 / 2)

if negative_count > len(negative):
    negative_count = len(negative)

negative = negative.sample(
    n=negative_count,
    random_state=42
)

new_test = pd.concat([positive, negative])

X_new_test = new_test.drop("target", axis=1)
y_new_test = new_test["target"]

y_probability = model.predict_proba(X_new_test)[:, 1]

y_pred = (y_probability >= 0.5).astype(int)

accuracy = accuracy_score(y_new_test, y_pred)
auc_score = roc_auc_score(y_new_test, y_probability)
loss = log_loss(y_new_test, y_probability)

print("Positive Class Percentage:", y_new_test.mean() * 100, "%")
print("Accuracy:", accuracy)
print("AUC:", auc_score)
print("Log Loss:", loss)