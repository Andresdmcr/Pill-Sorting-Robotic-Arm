import cv2 
import numpy as np
import joblib
import serial
import serial.tools.list_ports
import time

# Define HSV color ranges for containers
COLOR_RANGES = {
    'holder_green': ((35, 50, 50), (85, 255, 255)),
    'container_orange': ((5, 100, 100), (15, 255, 255)),
    'container_white': ((92, 0, 149), (137, 49, 255)),  # Previously updated values for white
    'container_blue': ((86, 191, 24), (180, 255, 255))  # Updated values for blue
}

PILL_TO_CONTAINER = {
    'pill_red': 'container_orange',
    'pill_white': 'container_white',
    'pill_silver': 'container_blue'
}

# Load the Random Forest model
model_path = r"C:\Users\andre\Downloads\pill_classifier.pkl"
try:
    clf, categories = joblib.load(model_path)
    print("Random Forest model and categories loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Function to extract color histogram features
def extract_color_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

# Function to classify a pill region
def classify_pill_region(roi):
    roi_resized = cv2.resize(roi, (64, 64))
    hist_features = extract_color_histogram(roi_resized).reshape(1, -1)
    probabilities = clf.predict_proba(hist_features)[0]
    predicted_class_index = np.argmax(probabilities)
    predicted_class = categories[predicted_class_index]
    confidence = probabilities[predicted_class_index]
    return predicted_class, confidence

# Function to create a mask for a specific HSV range
def create_color_mask(frame, lower_hsv, upper_hsv):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_bound = np.array(lower_hsv)
    upper_bound = np.array(upper_hsv)
    return cv2.inRange(hsv_frame, lower_bound, upper_bound)

# Function to find the center of the largest contour in the mask
def find_object_position(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy), largest_contour
    return None, None

# Function to detect containers in the frame
def detect_containers(frame):
    containers = {}
    for name, (lower, upper) in COLOR_RANGES.items():
        if 'container' in name:
            mask = create_color_mask(frame, lower, upper)
            position, contour = find_object_position(mask)
            if position:
                containers[name] = (position, contour)
    return containers

# Function to determine relative positions of containers
def determine_relative_positions(containers):
    positions = {}
    sorted_containers = sorted(containers.items(), key=lambda item: item[1][0][0])  # Sort by x-coordinate
    if len(sorted_containers) == 3:
        positions[sorted_containers[0][0]] = "Left"
        positions[sorted_containers[1][0]] = "Up"
        positions[sorted_containers[2][0]] = "Right"
    return positions

try:
    clf, categories = joblib.load(model_path)
    print("Random Forest model and categories loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize the Arduino connection
try:
    ser = serial.Serial('COM9', 9600)  # Replace 'COM9' with your Arduino port
    time.sleep(2)  # Wait for the connection to initialize
    print("Connected to Arduino.")
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    exit()

# Function to send a command to the Arduino
def send_command_to_arduino(position):
    command = position[0].upper()  # Use the first letter of the position (e.g., 'L', 'R', 'U')
    ser.write((command + '\n').encode())
    print(f"Command sent to Arduino: {command}")

    # Wait for Arduino to complete the task
    try:
        print("Waiting for Arduino to respond...")
        start_time = time.time()
        while True:
            if ser.in_waiting > 0:  # Check if there is data in the buffer
                response = ser.readline().decode().strip()
                if response == "Done":
                    print("Arduino has completed the task.")
                    time.sleep(2)  # Wait 2 seconds before processing the next pill
                    break
                else:
                    print(f"Unexpected response from Arduino: {response}")
            if time.time() - start_time > 60:  # Timeout after 60 seconds
                print("Timeout waiting for Arduino. Moving to the next pill.")
                break
    except Exception as e:
        print(f"Error communicating with Arduino: {e}")

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # Use MJPEG streaming
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_detected_pill = None
    confidence_threshold = 0.6
    frame_count = 0  # Counter for consecutive frames with the same pill
    action_taken = False  # To prevent multiple actions for the same pill

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        # Detect the green holder (assumes the pill is placed on it)
        green_mask = create_color_mask(frame, *COLOR_RANGES['holder_green'])
        green_position, green_contour = find_object_position(green_mask)

        if green_position and green_contour is not None:
            x, y, w, h = cv2.boundingRect(green_contour)
            roi = frame[y:y+h, x:x+w]

            # Classify the pill
            predicted_class, confidence = classify_pill_region(roi)

            # Normalize the predicted_class to lowercase and match dictionary format
            predicted_class = predicted_class.strip().lower()  # Normalize input
            predicted_class = f"pill_{predicted_class}"  # Ensure format matches dictionary keys

            # Only process if confidence is above the threshold
            if confidence >= confidence_threshold:
                if predicted_class == last_detected_pill:
                    frame_count += 1
                else:
                    frame_count = 1  # Reset frame count for a new pill
                    last_detected_pill = predicted_class
                    action_taken = False  # Allow action for the new pill
                    print(f"Detected Pill: {predicted_class.replace('pill_', '').capitalize()} with Confidence: {confidence:.2f}")

                if frame_count >= 5 and not action_taken:
                    # Find the container for the pill
                    containers = detect_containers(frame)
                    relative_positions = determine_relative_positions(containers)

                    # Ensure we stop printing after action is taken
                    target_container = PILL_TO_CONTAINER.get(predicted_class)
                    if target_container:
                        if target_container in relative_positions:
                            target_position = relative_positions[target_container]
                            print(f"Pill {predicted_class.replace('pill_', '').capitalize()} -> Container Location: {target_position}")
                            send_command_to_arduino(target_position)
                            action_taken = True  # Action performed for the current pill
                        else:
                            print(f"Error: Target container '{target_container}' not found in relative positions!")
                    else:
                        print(f"Error: No container mapping found for pill {predicted_class}")
            else:
                frame_count = 0  # Reset frame count on low confidence

            # Draw bounding box and label for the detected pill
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{predicted_class.replace('pill_', '').capitalize()} ({confidence:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Detect and label containers on the frame
        containers = detect_containers(frame)
        for name, (position, contour) in containers.items():
            cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)
            cv2.putText(frame, name, (position[0] - 20, position[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow("Pill and Container Detection", frame)

        # Exit gracefully on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
