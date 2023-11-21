import cv2
import os 
import numpy as np
import CPR_tools as cpr
from tqdm import tqdm

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



if __name__ == "__main__":
    
    DISPLAY_IMAGE_PATH ='C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/images/realTest_v3/unprocessed/WIN_20231113_12_53_28_Pro.jpg'
    # PREDICTION_FILE_PATH ='C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/runs/detect/predict5/labels/WIN_20231113_12_53_28_Pro.txt'
    PREDICTION_FILE_PATH = 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/output/hough/WIN_20231113_12_53_28_Pro.txt'
    DISPLAY_TIME = 5000
    discriminate(PREDICTION_FILE_PATH, DISPLAY_IMAGE_PATH, DISPLAY_TIME)
    showPredictions(good_colony_file_path='C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/output/good_colonies/WIN_20231113_12_53_28_Pro.txt', 
                    bad_colony_file_path= 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/output/bad_colonies/WIN_20231113_12_53_28_Pro.txt', 
                    image_path=DISPLAY_IMAGE_PATH,
                    display_time=DISPLAY_TIME)
    showColonies('C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/output/good_colonies/WIN_20231113_12_53_28_Pro.txt', DISPLAY_IMAGE_PATH, margin=2)
    # showPredictions(good_colony_file_path= './output/hough/WIN_20231113_12_53_28_Pro.txt', image_path = 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/images/realTest_v3/unprocessed/WIN_20231113_12_53_28_Pro.jpg')



