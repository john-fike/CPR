import cv2
import numpy as np

def imextendedmin(img, h):
    marker = img + h
    h_min = cv2.erode(marker, h)
    return (img == h_min)

def imimposemin(img, minima):
    marker = np.full_like(img, np.inf)
    marker[minima] = 0
    mask = np.minimum((img + 1), marker)
    return cv2.erode(marker, mask)

# Load your image
image = cv2.imread('sethTestImages/processed/2-13-18_10^0.jpg', 0)  # Load the image in grayscale

# Create the marker image
H = np.zeros(image.shape, dtype=np.uint8)

# Find the extended minima
extended_minima = imextendedmin(image, H)

# Impose the minima
imposed_minima = imimposemin(image, extended_minima)

# Convert extended_minima to a format suitable for display
extended_minima_display = (extended_minima * 255).astype(np.uint8)

# Show the extended_minima image
cv2.imshow('Extended Minima', extended_minima_display)

# Wait for a key event and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
