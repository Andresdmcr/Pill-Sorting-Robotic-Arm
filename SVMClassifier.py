import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Function to extract color histogram features
def extract_color_histogram(image):
    """Extracts color histogram features in HSV space."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

# Function to load and preprocess images
def load_images(data_dir, img_size=(64, 64)):
    """Loads images, extracts features, and assigns labels."""
    labels = []
    features = []
    categories = os.listdir(data_dir)

    for label, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            img = cv2.imread(file_path)
            if img is not None:
                img_resized = cv2.resize(img, img_size)
                hist_features = extract_color_histogram(img_resized)
                features.append(hist_features)
                labels.append(label)

    return np.array(features), np.array(labels), categories

# Main function for training and saving the model
def main():
    # Load the dataset
    data_dir = r"C:\Users\andre\Downloads\PillColors"  # Path to your dataset
    print("Loading images...")
    X, y, categories = load_images(data_dir)
    print(f"Loaded {len(X)} images from {len(categories)} categories.")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the SVM classifier
    print("Training the SVM model...")
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the SVM model...")
    y_pred = svm_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=categories))

    # Save the model and category labels
    model_path = r"C:\Users\andre\Downloads\pill_svm_model.pkl"
    joblib.dump((svm_model, categories), model_path)
    print(f"Model and categories saved to {model_path}")

# Run the main function
if __name__ == "__main__":
    main()
