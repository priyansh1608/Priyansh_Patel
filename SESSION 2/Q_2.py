"""
Q - 2 : Split the loaded Spotify dataset into training and test sets using sklearn's train_test_split function, 
        with 80% for training and 20% for testing. Print the number of rows in each set.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("SESSION 2\spotify_top_50.csv")

X = df[
    [
        "Danceability",
        "Energy",
        "Loudness",
        "Speechiness",
        "Acousticness",
        "Instrumentalness",
        "Liveness",
        "Valence",
        "Tempo"
    ]
]

y = df["Popularity"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))