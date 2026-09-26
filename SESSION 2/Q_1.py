"""
Q - 1 : Download the 'Spotify Top 50 Songs' dataset from Kaggle, load it into a pandas DataFrame, and 
        identify which columns should be used as features and which as the label if you want to predict 
        a song's popularity score.
"""


import pandas as pd

df = pd.read_csv("spotify_top_50.csv")

print("First 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

features = [
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

label = "Popularity"

print("\nFeatures:")
print(features)

print("\nLabel:")
print(label)