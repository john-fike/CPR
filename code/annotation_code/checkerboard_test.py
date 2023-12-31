import cv2
import numpy as np

# Load the image in grayscale
image = cv2.imread('C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/chekcerboard.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if image is not None:
    # Get image dimensions
    height, width = image.shape

    # Create a new image to store the modified pixels
    modified_image = np.copy(image)

    # Define the intensity change for lighter and darker pixels
    light_intensity_change = 30
    dark_intensity_change = -30

    # Iterate through each pixel and modify intensity
    for y in range(0, height, 2):
        for x in range(0, width, 2):
            modified_image[y, x] = np.clip(modified_image[y, x] + light_intensity_change, 0, 255)
            
    for y in range(1, height, 2):
        for x in range(1, width, 2):
            modified_image[y, x] = np.clip(modified_image[y, x] + dark_intensity_change, 0, 255)

    # Display the original and modified images
    cv2.imshow('Original Image', image)
    cv2.imshow('Modified Image', modified_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/chekcerboard_modified.jpg', modified_image)

else:
    print('Image not loaded.')
