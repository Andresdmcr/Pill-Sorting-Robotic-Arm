import cv2
import os

# Set the folder where screenshots will be saved
save_folder = r"C:\Users\andre\Downloads\PillColors\white"

# Ensure the save folder exists
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Callback function to handle mouse events
def draw_bounding_box(event, x, y, flags, param):
    global start_point, end_point, cropping, frame, count

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

        # Take a screenshot of the selected area
        if start_point and end_point:
            x1, y1 = start_point
            x2, y2 = end_point
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            # Crop the selected area
            roi = frame[y_min:y_max, x_min:x_max]

            # Save the cropped area as an image
            screenshot_path = os.path.join(save_folder, f"white_{count}.jpg")
            cv2.imwrite(screenshot_path, roi)
            print(f"Screenshot saved: {screenshot_path}")
            count += 1

# Initialize variables
start_point = None
end_point = None
cropping = False
count = 1

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

cv2.namedWindow("Webcam")
cv2.setMouseCallback("Webcam", draw_bounding_box)

print("Press 'q' to quit.")

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
