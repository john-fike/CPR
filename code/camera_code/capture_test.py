import cv2

# Open the first connected camera
cap = cv2.VideoCapture(1)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera")
    exit()

# Set the exposure value (adjust the value as needed)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Example value (-6 for lower exposure)

# Capture a frame from the camera
ret, frame = cap.read()

# Check if the frame was captured successfully
if not ret:
    print("Error: Could not read a frame")
    exit()

# Save the captured frame as an image
cv2.imwrite('C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/code/camera_code/test_image.jpg', frame)

# Release the camera
cap.release()

# Load an image
image = cv2.imread('C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/code/camera_code/test_image.jpg')

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load the image")
    exit()

# Example: Increase brightness (simulate higher exposure)
increased_brightness = cv2.add(image, 50)  # Adjust the value to change brightness

# Example: Decrease brightness (simulate lower exposure)
decreased_brightness = cv2.subtract(image, 50)  # Adjust the value to change brightness

# Display the original and adjusted images
cv2.imshow('Original Image', image)
cv2.imshow('Increased Brightness (Higher Exposure)', increased_brightness)
cv2.imshow('Decreased Brightness (Lower Exposure)', decreased_brightness)
cv2.waitKey(0)
cv2.destroyAllWindows()
