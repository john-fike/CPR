import cv2

# Initialize the webcam
cap = cv2.VideoCapture(1)  # 0 represents the default camera, change if necessary

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set exposure value (0 for automatic, positive values for manual control)
exposure_value = -20  # You can adjust this value as needed

# Set the exposure property
cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame is captured successfully
    if not ret:
        print("Error: Could not capture frame.")
        break

    # Display the frame in a window
    cv2.imshow("Webcam Stream", frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
