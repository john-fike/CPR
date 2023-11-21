import cv2
import numpy as np
import os
import skimage.morphology as morph
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def showColonies(prediction_file_path, image_path, display_time = 500, margin = 1):
    # -----------------------------------------------LOAD IMAGE AND PROPERTIES------------------
    img = cv2.imread(image_path)
    # Check if the image was loaded successfully
    if img is None:
        print("Error: Could not read image file")
        exit()
    img_width = img.shape[1]
    img_height = img.shape[0]

    with open(prediction_file_path) as file:
        colony_lines = file.readlines()
        for colony_line in colony_lines:
            elements = colony_line.split()
            x = int(float(elements[1]) * img_width)
            y = int(float(elements[2]) * img_height)
            w = int(float(elements[3]) * img_width * margin)
            h = int(float(elements[4]) * img_height * margin)

            # -----------------------------------------------CROP & GRAY---------------------------------
            cropped_img = img[int(y-h) : int(y+h) , int(x-w) : int(x+w)]
            if cropped_img is None:
                print("Error: Could not crop image")
                exit()
            
            display_img = cv2.resize(cropped_img, (640, 640))  
            cv2.imshow('Colony', display_img)
            cv2.waitKey(display_time)
            cv2.destroyAllWindows()


def discriminate(PREDICTION_FILE_PATH, 
                 image_file_path,
                 PISS = 'FUCK',      #do not remove--code will break
                 GOOD_OUTPUT_PATH = 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/output/good_colonies/',
                 BAD_OUTPUT_PATH  = 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/output/bad_colonies/',
                 MIN_DISTANCE = .03,
                 MIN_SELECTION_CONFIDENCE = 0.14, 
                 MIN_DISCRIMINATION_CONFIDENCE = .05, 
                 MIN_SIZE = .01, 
                 MAX_SIZE = .5, MAXIMUM_RATIO = .15, 
                 PETRI_DISH_RADIUS = .4,
                 ):

    base_file_name = os.path.splitext(os.path.basename(PREDICTION_FILE_PATH))[0]
    good_file_name = os.path.join(str(GOOD_OUTPUT_PATH), base_file_name + '.txt')
    bad_file_name  = os.path.join(str(BAD_OUTPUT_PATH),  base_file_name + '.txt')
    print("Base file name: " , base_file_name)
    print("Good file name: " , good_file_name)
    print("Bad file name: "  , bad_file_name)

    good_colonies = []
    bad_colonies = []

    #clear files
    with open(good_file_name, 'w') as good_file:
        pass
    with open(bad_file_name, 'w') as bad_file:
        pass

    with open(PREDICTION_FILE_PATH) as predictionFile:
        lines = predictionFile.readlines()
        for main_colony_line in tqdm(lines):
            main_colony   = main_colony_line.split() # [class, x, y, width, height, confidence]
            main_colony_x = float(main_colony[1])
            main_colony_y = float(main_colony[2])
            main_colony_w = float(main_colony[3])
            main_colony_h = float(main_colony[4])
            main_colony_confidence = float(main_colony[5])
            ratio = abs((float(main_colony[4]) / float(main_colony[3])) - 1 ) #ok really this is how not square it is not the ratio but close enough
            
            is_bad_colony = True
            if(cpr.distance(x0=main_colony_x, y0=main_colony_y) < PETRI_DISH_RADIUS and main_colony_confidence > MIN_SELECTION_CONFIDENCE and 
               cpr.binary_disciminate(img_file_path=image_file_path, x=main_colony_x, y=main_colony_y, width=main_colony_w,
                                      height=main_colony_h, original_display=False, bad_display=False, display_time=500, margin=2)
               ):
                is_bad_colony = False
                for neighbor_colony_line in lines: 
                    neighbor_colony = neighbor_colony_line.split()
                    neighbor_colony_x = float(neighbor_colony[1])
                    neighbor_colony_y = float(neighbor_colony[2])
                    neighbor_colony_r = float(neighbor_colony[3])
                    neighbor_colony_confidence = float(neighbor_colony[5])

                    distance_between_colony_centers = cpr.distance(x0=main_colony_x, y0=main_colony_y, r0=main_colony_w, 
                                                                   x1=neighbor_colony_x, y1=neighbor_colony_y, r1=neighbor_colony_r)

                    if (distance_between_colony_centers <  MIN_DISTANCE and          #distance to colony
                        distance_between_colony_centers != 0.0 and                   #make sure it's not the same colony
                        neighbor_colony_confidence > MIN_DISCRIMINATION_CONFIDENCE): #make sure the colony prediction is confident enough to be used for discrimination
                        is_bad_colony = True

            if is_bad_colony:
                if not bad_colonies.__contains__(main_colony_line):
                    with open(bad_file_name, 'a') as bad_file:
                        bad_file.write(main_colony_line)
                        bad_colonies.append(main_colony_line)

            else:
                if not good_colonies.__contains__(main_colony_line):
                    with open(good_file_name, 'a') as good_file:
                        good_file.write(main_colony_line)
                        lines.remove(main_colony_line)
                        
def showPredictions(good_colony_file_path=None, bad_colony_file_path=None, image_path=None, display_time = 2000):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the image")
        exit()
    
    if good_colony_file_path is not None:
        with open(good_colony_file_path) as good_colony_file:
            good_colonies = good_colony_file.readlines()
            for colony_line in good_colonies:
                if colony_line is not None:
                    elements = colony_line.split()
                    x = int(float(elements[1]) * image.shape[1])
                    y = int(float(elements[2]) * image.shape[0])
                    r = int(float(elements[3]) * image.shape[1] / 2)
                    cv2.circle(image, (x, y), r, (0, 255, 0), 1)

    if bad_colony_file_path is not None:
        with open(bad_colony_file_path) as bad_colony_file:
            bad_colonies = bad_colony_file.readlines()
            for colony_line in bad_colonies:
                if colony_line is not None:
                    elements = colony_line.split()
                    x = int(float(elements[1]) * image.shape[1])
                    y = int(float(elements[2]) * image.shape[0])
                    r = int(float(elements[3]) * image.shape[1] / 2)
                    cv2.circle(image, (x, y), r, (0, 0, 255), 1)
    
    image = cv2.resize(image, (640, 640))
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

