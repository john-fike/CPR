import cv2
import numpy as np
import os

def add_hough_circles(image, prediction_file, display=False, display_time=5000):
    if image is None:
        print("Error: Could not load the image")
        exit()

    # check if image is grayscale 
    if len(image.shape) > 2:
    # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (3, 3), 0)                                                                               ##PARAM                  

    # Apply Canny edge detector
    edges = cv2.Canny(blurred, 50, 60)                                                                                        ##PARAM

    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=15, minRadius=5, maxRadius=30)   ##PARAM
    
    #clear contents of output.txt
    open(prediction_file, 'w').close()

    #plot circles
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if display: 
                cv2.circle(image, (x, y), r, (255, 0, 0), 1)
            image_width = image.shape[1]
            image_height = image.shape[0]    
            with open (prediction_file, 'a') as f:
                f.write("0 " + str(x/image_width) + " " + str(y/image_height) + " " + str(r/image_width) +  " " + str(r/image_width) + " .07" "\n")

    else:
        print("No circles detected")

    if display:
        image = cv2.resize(image, (640, 640))
        cv2.imshow('Result', image)
        cv2.waitKey(display_time)
        cv2.destroyAllWindows()


#iterate through a foler
if __name__ == "__main__":
    filepath = './images/realTest_v3/'
    unprocessedFiles = os.path.join(filepath, 'unprocessed')
    processedFiles = os.path.join(filepath, 'processed')

    for file in os.listdir(unprocessedFiles):
        # read in image and convert to grayscale
        img = cv2.imread(os.path.join(unprocessedFiles, file) , cv2.IMREAD_GRAYSCALE)
        add_hough_circles(img, prediction_file= './output.txt' , display=True)
