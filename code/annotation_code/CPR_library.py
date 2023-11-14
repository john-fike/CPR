import cv2
import numpy as np
import os
import skimage.morphology as morph
from scipy.stats import norm
import matplotlib.pyplot as plt


def add_hough_circles(image, prediction_file_path, display=False, display_time=5000):
    if image is None:
        print("Error: Could not load the image")
        exit()

    # make image grayscale if needed 
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (3, 3), 0)                                                                               ##PARAM                  

    # Apply Canny edge detector
    edges = cv2.Canny(blurred, 50, 60)                                                                                        ##PARAM

    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=15, minRadius=5, maxRadius=30)   ##PARAM
    
    #clear contents of output.txt
    open(prediction_file_path, 'w').close()

    #plot circles
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if display: 
                cv2.circle(image, (x, y), r, (255, 0, 0), 1)
            image_width = image.shape[1]
            image_height = image.shape[0]    
            with open (prediction_file_path, 'a') as f:
                f.write("0 " + str(x/image_width) + " " + str(y/image_height) + " " + str(r/image_width) +  " " + str(r/image_width) + " .07" "\n")
    else:
        print("No hough circles detected")

    if display:
        image = cv2.resize(image, (640, 640))
        cv2.imshow('Result', image)
        cv2.waitKey(display_time)
        cv2.destroyAllWindows()


#function to determine the distance from the center of the image to the center of the box
def distance(x0, y0, r1=0, x1=.5, y1=.5, r2=0):
    x_dist = abs(x0 - x1) ** 2
    y_dist = abs(y0 - y1) ** 2
    distance = (x_dist + y_dist) ** .5
    distance = distance - (r1 + r2)
    return (x_dist + y_dist) ** .5

def binary_disc (img_file_path, x, y, width, height, MARGIN = 1, erosion_thresholds = (130, 110, 90, 60), erosion_iterations = (0, 1, 2, 3), display = False):

    img = cv2.imread(img_file_path)
    # Check if the image was loaded successfully
    if img is None:
        print("Error: Could not read image file")
        exit()
    # -----------------------------------------------CROP IMAGE----------------------
    cropped_image = img[int(y-height) : int(y+height) , int(x-width) : int(x+width)]
    if(cropped_image is None):
        print("Error: Could not crop image")
        exit()
    # -----------------------------------------------CIRCLE DETECT----------------------
    else:
        gray_cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        erosion_img = cv2.erode(gray_cropped_image, np.ones((2, 2), np.uint8), iterations=5) 

        # -----------------------------------------------THRESHOLD BINERIZATION----------------------
        hist = cv2.calcHist([gray_cropped_image], [0], None, [256], [0, 256])
        hist = hist.ravel()
        x = np.linspace(0, 255, 256)
        param = norm.fit(x, loc=np.mean(hist), scale=np.std(hist))
        mean, std_dev = param
        k = .5 
        threshold = int(mean - k * std_dev)
        binary_image = cv2.threshold(gray_cropped_image, threshold, 255, cv2.THRESH_BINARY)[1]
        binary_image = cv2.bitwise_not(binary_image)                                                #invert                                 
        
        # -----------------------------------------------EROSION----------------------
        for l in len(erosion_iterations):
            iterations = erosion_iterations[l]
            erosion_threshold = erosion_thresholds[l]
            eroded_binary_image = cv2.erode(binary_image, np.ones((2,2), np.uint8), iterations=iterations)
            binary_image_sum = np.sum(eroded_binary_image)          #sum of all pixels in the image
            binary_image_sum = binary_image_sum / (width * height)  #normalize
            if binary_image_sum > erosion_threshold:
                print("FUCK!!!")
            if display:
                title = "Erosion iteration: " + iterations + "Normalized intensity: " + binary_image_sum + "Threshold: " + erosion_threshold
                cv2.imshow(title, eroded_binary_image)
            
        
        # eroded_binary_image = cv2.erode(binary_image, np.ones((2,2), np.uint8), iterations=10)
    # very_eroded_binary_image = cv2.erode(binary_image, np.ones((2,2), np.uint8), iterations=20)
