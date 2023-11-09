import cv2
import os 
import numpy as np

#function to determine the distance from the center of the image to the center of the box
def distance(x0, y0, x1=.5, y1=.5):
    x_dist = abs(x0 - x1) ** 2
    y_dist = abs(y0 - y1) ** 2
    return (x_dist + y_dist) ** .5

DISPLAY_IMAGE_FOLDER_PATH = 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/images/realTest_v2/processed/'
PREDICTION_FOLDER_PATH = 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/runs/detect/predict3/'

# #DEFAULTS
# DISPLAY_TIME = 10000
# MIN_DISTANCE = .02
# MIN_SELECTION_CONFIDENCE = 0.0
# MIN_DISCRIMINATION_CONFIDENCE = .15       #if a colony is next to another colony, the colony it's next to must have a confidence greater than this to be displayed  #maybe i should make this a percentage of the average confidence of all the predictions in the image (minus those that are outside of the dish)
# MIN_SIZE = .01
# MAX_SIZE = .5
# MAXIMUM_RATIO = .15    #ratio of height to width of box. If the ratio is greater than this, the box is not displayed. ratio is calculated as the absolute value of 1 - (height / width)

#UNCOMMENT TO DISABLE SELECTION VARIABLES
DISPLAY_TIME = 5000
MIN_DISTANCE = 0.0 
MIN_SELECTION_CONFIDENCE = 0.0 
MIN_DISCRIMINATION_CONFIDENCE = 0.0
MIN_SIZE = 0.0
MAX_SIZE = 1.0
MAXIMUM_RATIO = .15    #ratio of height to width of box. If the ratio is greater than this, the box is not displayed. ratio is calculated as the absolute value of 1 - (height / width)

labelFolderPath = os.path.join(PREDICTION_FOLDER_PATH, 'labels')
#array that holds the average values for 
#iterates through all files in PREDICTION_FOLDER_PATH
#goes throug all the annotations (lines) in the file and displays a circle around the colony if it meets the criteria
#you specifed above 
for file in os.listdir(PREDICTION_FOLDER_PATH):
    if file.endswith('.jpg'):
        img = cv2.imread(os.path.join(DISPLAY_IMAGE_FOLDER_PATH, file))

        print(os.path.join(DISPLAY_IMAGE_FOLDER_PATH, file))
        if img is None:
            print("Error: Could not load the image")
            exit()

        file_name = os.path.splitext(file)[0]
        with open(os.path.join(labelFolderPath, file_name + ".txt")) as annotationFile:
            lines = annotationFile.readlines()
            for line in lines:
                elements = line.split() #elements[] = [class, x, y, width, height, confidence]
                #determine height 
                ratio = abs((float(elements[4]) / float(elements[3])) - 1 )
                print()
                #checks: colony is within petri dish     AND     colony is large enough     AND     colony is small enough    AND    ratio is small enough (colony is roound enough)
                # if (distance(float(elements[1]),float(elements[2])) < .4) and float(elements[4]) > MIN_SIZE and float(elements[4]) < MAX_SIZE and ratio < MAXIMUM_RATIO:
                if(distance(float(elements[1]),float(elements[2])) < .4 and float(elements[5]) > MIN_SELECTION_CONFIDENCE):
                    bad = False
                    for line in lines: 
                        more_elements = line.split()
                        #assess the closest colony to the current colony, if it is closer than MIN_DISTANCE, don't display the current colony
                        if (distance(float(more_elements[1]),float(more_elements[2]), float(elements[1]), float(elements[2])) < MIN_DISTANCE and 
                            distance(float(more_elements[1]),float(more_elements[2]), float(elements[1]), float(elements[2])) != 0.0 and 
                            float(elements[5]) > MIN_DISCRIMINATION_CONFIDENCE):
                            bad = True
                    if not bad:
                        #if the colony meets all the criteria, display it  
                        x = int(float(elements[1]) * img.shape[1])
                        y = int(float(elements[2]) * img.shape[0])
                        w = int(float(elements[3]) * img.shape[1] / 2)
                        h = int(float(elements[4]) * img.shape[0] / 2)
                        color = int(float(elements[5]) * 255)

                        # x = int(float(elements[1]) * img.shape[1])
                        # y = int(float(elements[2]) * img.shape[0])
                        # w = int(float(elements[3]) * img.shape[1])
                        # h = int(float(elements[4]) * img.shape[0])
                        # cropped_image = img[y:y+h, x:x+w]

                        # # Convert the cropped image to grayscale
                        # gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

                        # # Apply Hough Circle Transform
                        # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                        #                         param1=50, param2=30, minRadius=0, maxRadius=0)

                        # if circles is not None:
                        #     # Convert the (x, y) coordinates and radius of the circles to integers
                        #     circles = np.round(circles[0, :]).astype("int")

                        #     # Draw the circles on the original image
                        #     for (x, y, r) in circles:
                        #         cv2.circle(img, (x + x, y + y), r, (0, 255, 0), 2)
                        cv2.circle(img, (x, y), w, (0, color, 0), 1)
                        cv2.circle(img, (x, y), int(w*1.3), (0, color, 0), 1)
                        #draw a dot at x, y on image
                        cv2.circle(img, (x, y), 1, (0, 0, 255), 1)

                #draw circle with radius 10 at center of image
            cv2.circle(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)), int(.5 * img.shape[0]), (0, 0, 255), 1)
            cv2.imshow('image', img)
            cv2.waitKey(DISPLAY_TIME)





