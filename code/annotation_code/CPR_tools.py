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
def distance(x0, y0, r0=0, x1=.5, y1=.5, r1=0):
    x_dist = abs(x0 - x1) ** 2
    y_dist = abs(y0 - y1) ** 2
    distance = (x_dist + y_dist) ** .5
    distance = distance - (r0 + r1)
    return (x_dist + y_dist) ** .5


#returns boolean of whether or not the colony is good 
#if the binerized image has a normalized intensity above 
# the threshold at any erosion iteration, the colony is bad
def binary_disciminate(img_file_path, x, y, width, height, margin = 1, erosion_thresholds = (180, 140, 110, 90), erosion_iterations = (0, 1, 2, 3), original_display = False, bad_display = False, good_display=False, display_time = 2000):

    print("binary shit!")
    # -----------------------------------------------LOAD IMAGE AND PROPERTIES------------------
    img = cv2.imread(img_file_path)
    # Check if the image was loaded successfully
    if img is None:
        print("Error: Could not read image file")
        exit()
    
    img_width = img.shape[1]
    img_height = img.shape[0]
    x = img_width * x
    y = img_height * y
    width = img_width * width * margin
    height = img_height * height * margin

    # -----------------------------------------------CROP & GRAY---------------------------------
    cropped_image = img[int(y-height) : int(y+height) , int(x-width) : int(x+width)]
    if cropped_image is None:
        print("Error: Could not crop image")
        exit()

    if len(cropped_image.shape) > 2:
        gray_cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_cropped_image = cropped_image

    # -----------------------------------------------THRESHOLD BINERIZATION----------------------
    hist = cv2.calcHist([gray_cropped_image], [0], None, [256], [0, 256])
    hist = hist.ravel()
    z = np.linspace(0, 255, 256)
    param = norm.fit(z, loc=np.mean(hist), scale=np.std(hist))
    mean, std_dev = param
    k = .5 
    threshold = int(mean - k * std_dev)
    binary_image = cv2.threshold(gray_cropped_image, threshold, 255, cv2.THRESH_BINARY)[1]
    binary_image = cv2.bitwise_not(binary_image)                                                #invert                                 
    
    if original_display:
        title = "Original"
        display_image = cv2.resize(cropped_image, (640, 640))
        cv2.imshow(title, display_image)
        cv2.waitKey(display_time)

    # -----------------------------------------------EROSION----------------------
    for l in range(len(erosion_iterations)):
        iterations = erosion_iterations[l]
        erosion_threshold = erosion_thresholds[l]
        eroded_binary_image = cv2.erode(binary_image, np.ones((2,2), np.uint8), iterations=iterations)
        binary_image_sum = np.sum(eroded_binary_image)          #sum of all pixels in the image
        binary_image_sum = binary_image_sum / (width * height)  #normalize
        if original_display:            
            title = "Erosion iteration: " + str(iterations) + " Normalized intensity: " + str(int(binary_image_sum)) + " Threshold: " + str(erosion_threshold)
            display_image = cv2.resize(eroded_binary_image, (640, 640))
            cv2.imshow(title, display_image)
            cv2.waitKey(display_time)
            cv2.destroyAllWindows()

        if binary_image_sum > erosion_threshold:
            print("Colony exceed threshold at erosion iteration: " + str(iterations) + " Normalized intensity: " + str(int(binary_image_sum)) + " Threshold: " + str(erosion_threshold))
            if(bad_display):
                cv2.circle(img, (int(x), int(y)), int(width/margin), (0, 0, 255), 1)
                img = cv2.resize(img, (640, 640))
                cv2.imshow("Violating Colony", img)
                cv2.waitKey(display_time)
                cv2.destroyAllWindows()
                display_image = cv2.resize(eroded_binary_image, (640, 640))
                cv2.imshow("BAD Colony", display_image)
                cv2.waitKey(display_time)
                cv2.destroyAllWindows()
            return False
        
        else:
            print("Colony passed threshold at erosion iteration: " + str(iterations) + " Normalized intensity: " + str(int(binary_image_sum)) + " Threshold: " + str(erosion_threshold))
            if(good_display):
                cv2.circle(img, (int(x), int(y)), int(width/margin), (0, 0, 255), 1)
                img = cv2.resize(img, (640, 640))
                cv2.imshow("GOOD Colony", img)
                cv2.waitKey(display_time)
                cv2.destroyAllWindows()
                display_image = cv2.resize(eroded_binary_image, (640, 640))
                cv2.imshow("GOOD Colony", display_image)
                cv2.waitKey(display_time)
                cv2.destroyAllWindows()
            return True

#creates a .txt file for each of the images in image_folder_path
#puts it in prediction_output_path
#margin is the multiplier for the radius of the circle. this is useful for binerization stuff, because
#yolo boxes tend to be bigger than the actual colony and hough circles tend to be smaller
#output_confidence is the confidence written to the .txt file. this is useful for showPrediction stuff 
#display is a boolean that determines whether or not to display the image with the hough circles
#display_time is the time in milliseconds that the image is displayed for
#hough_confidence is the "confidence" threshold for the hough circles
#PARAM marks something else you can change if shit isn't working 
def add_hough_circles(image_path, 
                      prediction_path, 
                      margin = 1, 
                      output_confidence = ".9", 
                      display=False, 
                      display_time=5000,
                      hough_confidence = 15
                      ):
    img = cv2.imread(os.path.join(image_path) , cv2.IMREAD_COLOR)
    if img is None:
        print("Error: Could not load the image")
        exit()

    # make image grayscale if needed 
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # detect circles
    blurred = cv2.GaussianBlur(img, (2, 2), 0)          #PARAM
    edges = cv2.Canny(blurred, 50, 60)                                                                                        ##PARAM
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=hough_confidence, minRadius=5, maxRadius=30)   ##PARAM
    
    #clear contents of output.txt
    open(prediction_path, 'w').close()

    #plot circles
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

    if display:
        img = cv2.imread(os.path.join(image_path) , cv2.IMREAD_COLOR)

    for (x, y, r) in circles:
        image_width = img.shape[1]
        image_height = img.shape[0]    
        with open (prediction_path, 'a') as f:
            f.write("0 " + str(x/image_width) + " " + str(y/image_height) + " " + str(r * margin/image_width) +  " " + str(r* margin/image_width) + " " + output_confidence + "\n")
        
        if display:               
            cv2.circle(img, (x, y), (r * margin), (255, 0, 0), 1)
    if display:
        img = cv2.resize(img, (640, 640))
        cv2.imshow('Result', img)
        cv2.waitKey(display_time)
        cv2.destroyAllWindows()

    else:
        print("No hough circles detected")
        # eroded_binary_image = cv2.erode(binary_image, np.ones((2,2), np.uint8), iterations=10)
    # very_eroded_binary_image = cv2.erode(binary_image, np.ones((2,2), np.uint8), iterations=20)
