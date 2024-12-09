import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def extract_color_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def load_images(data_dir):
    labels = []
    features = []
    categories = os.listdir(data_dir)

    for label, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            img = cv2.imread(file_path)
            if img is not None:
                img_resized = cv2.resize(img, (64, 64))
                hist_features = extract_color_histogram(img_resized)
                features.append(hist_features)
                labels.append(label)

    return np.array(features), np.array(labels), categories

# Train and save model
data_dir = r"C:\Users\andre\Downloads\PillColors"

X, y, categories = load_images(data_dir)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

clf = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=categories))

# Save the model and categories
model_path = r"C:\Users\andre\Downloads\pill_classifier.pkl"
joblib.dump((clf, categories), model_path)
print(f"Model and categories saved to {model_path}")

