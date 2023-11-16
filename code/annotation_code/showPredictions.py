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
                 MIN_DISTANCE = .01,
                 MIN_SELECTION_CONFIDENCE = 0.15, 
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
            main_colony = main_colony_line.split() # [class, x, y, width, height, confidence]
            main_colony_x = float(main_colony[1])
            main_colony_y = float(main_colony[2])
            main_colony_w = float(main_colony[3])
            main_colony_h = float(main_colony[4])
            main_colony_confidence = float(main_colony[5])
            ratio = abs((float(main_colony[4]) / float(main_colony[3])) - 1 ) #ok really this is how not square it is not the ratio but close enough
            
            is_bad_colony = True
            if(cpr.distance(x0=main_colony_x, y0=main_colony_y) < PETRI_DISH_RADIUS and main_colony_confidence > MIN_SELECTION_CONFIDENCE and 
               cpr.binary_disciminate(img_file_path=image_file_path, x=main_colony_x, y=main_colony_y, width=main_colony_w, height=main_colony_h, bad_display=True, display_time=100)
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
                        
def showPredictions(good_colony_file_path, bad_colony_file_path, image_path, display_time = 2000):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the image")
        exit()
    
    with open(good_colony_file_path) as good_colony_file:
        good_colonies = good_colony_file.readlines()
        for colony_line in good_colonies:
            if colony_line is not None:
                elements = colony_line.split()
                x = int(float(elements[1]) * image.shape[1])
                y = int(float(elements[2]) * image.shape[0])
                r = int(float(elements[3]) * image.shape[1] / 2)
                cv2.circle(image, (x, y), r, (0, 255, 0), 1)

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
    

if __name__ == "__main__":
    
    DISPLAY_IMAGE_PATH ='C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/images/realTest_v3/unprocessed/WIN_20231113_12_53_28_Pro.jpg'
    PREDICTION_FILE_PATH ='C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/runs/detect/predict5/labels/WIN_20231113_12_53_28_Pro.txt'
    DISPLAY_TIME = 5000
    discriminate(PREDICTION_FILE_PATH, DISPLAY_IMAGE_PATH, DISPLAY_TIME)
    showPredictions(good_colony_file_path='C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/output/good_colonies/WIN_20231113_12_53_28_Pro.txt', 
                    bad_colony_file_path= 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/output/bad_colonies/WIN_20231113_12_53_28_Pro.txt', 
                    image_path=DISPLAY_IMAGE_PATH,
                    display_time=DISPLAY_TIME
                    )





    
    # for file in os.listdir(PREDICTION_FOLDER_PATH):
    #     if file.endswith('.jpg'):
    #         img = cv2.imread(os.path.join(DISPLAY_IMAGE_FOLDER_PATH, file))

    #         print(os.path.join(DISPLAY_IMAGE_FOLDER_PATH, file))
    #         if img is None:
    #             print("Error: Could not load the image")
    #             exit()

    #         file_name = os.path.splitext(file)[0]

        #         x = int(float(elements[1]) * img.shape[1])
        #         y = int(float(elements[2]) * img.shape[0])
        #         w = int(float(elements[3]) * img.shape[1] / 2)
        #         h = int(float(elements[4]) * img.shape[0] / 2)

                
        #         if not bad:
        #             #if the colony meets all the criteria, display it  
        #             color = (0, int(float(elements[5]) * 255), 0)

        #         if bad:
        #             color = (0, 0, 255)
        #             #plot line from x y to next_closest_colony_x next_closest_colony_y
        #             cv2.line(img, (x, y), (int(float(next_closest_colony_x)), int(float(next_closest_colony_y))), color, 1)
        #             cv2.circle(img, (next_closest_colony_x, next_closest_colony_y), int(next_closest_colony_r*1.3), color, 1)
                    
        #         # Plot center 
        #         cv2.circle(img, (x, y), int(w*1.3), color, 1)
        #         #draw a dot at x, y on image
        #         cv2.circle(img, (x, y), 1, (0, 0, 255), 1)

        #     #draw circle with radius 10 at center of image
        # cv2.circle(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)), int(PETRI_DISH_RADIUS * img.shape[0]), (0, 0, 255), 1)
        # #resize image to 640x640
        # img = cv2.resize(img, (640, 640))
        # cv2.imshow('image', img)
        # cv2.waitKey(DISPLAY_TIME)


                        # next_closest_colony_x = int(float(neighbor_colony[1]) * img.shape[1])
                        # next_closest_colony_y = int(float(neighbor_colony[2]) * img.shape[0])
                        # next_closest_colony_r = int(float(neighbor_colony[3]) * img.shape[1] / 2) 

