import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

bat_images = np.random.randint(0, 256, (10, 64, 64))
football_images = np.random.randint(0, 256, (10, 64, 64))

X = np.concatenate([bat_images, football_images])
y = np.array(
    ["cricket_bat"] * 10 + ["football"] * 10
)

X = X.reshape(20, -1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("Actual Labels:")
print(y_test)

print("\nPredicted Labels:")
print(prediction)

print("\nAccuracy:")
print(accuracy_score(y_test, prediction))

new_image = np.random.randint(0, 256, (1, 64, 64))
new_image = new_image.reshape(1, -1)

result = model.predict(new_image)

print("\nNew Image Prediction:")
print(result[0])

print("\nChanges Made:")
print("Random pixel arrays were used instead of downloading an image dataset.")
print("Images were reshaped into one-dimensional feature vectors before training.")