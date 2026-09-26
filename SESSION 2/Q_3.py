"""
Q - 3 : Given a simple linear regression model predicting song popularity from danceability, intentionally 
        use only 5% of the data for training and 95% for testing. Observe the model's performance and explain
        whether this is likely to cause underfitting or overfitting.<br><br><em><strong>Hint:</strong> Look at 
        the accuracy or error on both train and test sets to support your answer.</em>
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("SESSION 2\spotify_top_50.csv")

X = df[["Danceability"]]

y = df["Popularity"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.95,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)

train_error = mean_squared_error(y_train, train_prediction)
test_error = mean_squared_error(y_test, test_prediction)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTraining Error:", train_error)
print("Testing Error:", test_error)

print("\nConclusion:")
print("Using only 5% of the data for training can cause UNDERFITTING.")
print("The model has very little data to learn the relationship between")
print("danceability and popularity.")

