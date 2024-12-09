import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Function to extract color histogram features
def extract_color_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

# Callback function for drawing the bounding box and classifying the selected area
def draw_bounding_box(event, x, y, flags, param):
    global start_point, end_point, cropping, frame, clf, categories

    # Record starting point on left mouse button down
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        cropping = True

    # Update end_point as the mouse moves
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            end_point = (x, y)

    # Finalize the bounding box on left mouse button up
    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        cropping = False

        if start_point and end_point:
            x1, y1 = start_point
            x2, y2 = end_point
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            # Crop the selected area
            roi = frame[y_min:y_max, x_min:x_max]

            # Resize the cropped area and classify it
            roi_resized = cv2.resize(roi, (64, 64))
            hist_features = extract_color_histogram(roi_resized).reshape(1, -1)
            prediction = clf.predict(hist_features)
            predicted_class = categories[prediction[0]]
            print(f"Predicted Class: {predicted_class}")

            # Display the prediction on the frame
            cv2.putText(frame, f"Predicted: {predicted_class}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Initialize variables
start_point = None
end_point = None
cropping = False

# Load the trained model
model_path = r"C:\Users\andre\Downloads\pill_classifier.pkl"
try:
    clf, categories = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

cv2.namedWindow("Webcam")
cv2.setMouseCallback("Webcam", draw_bounding_box)

print("Drag to create a bounding box for classification. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Display the frame with the current bounding box
    temp_frame = frame.copy()
    if start_point and end_point and cropping:
        cv2.rectangle(temp_frame, start_point, end_point, (0, 255, 0), 2)

    cv2.imshow("Webcam", temp_frame)

    # Quit the program when 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()