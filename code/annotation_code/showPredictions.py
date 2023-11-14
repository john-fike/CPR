import cv2
import os 
import numpy as np

#determines distance between two scaled points. defaults to finding distance from center if provided with just one point
def distance(x0, y0, x1=.5, y1=.5):
    x_dist = abs(x0 - x1) ** 2
    y_dist = abs(y0 - y1) ** 2
    return (x_dist + y_dist) ** .5


def discriminate(PREDICTION_FILE_PATH, 
                 PISS = 'FUCK',
                 GOOD_OUTPUT_PATH = 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/output/good_colonies/',
                 BAD_OUTPUT_PATH = 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/output/bad_colonies/',
                 MIN_DISTANCE = .02,
                 MIN_SELECTION_CONFIDENCE = 0.0, 
                 MIN_DISCRIMINATION_CONFIDENCE = .15, 
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


    with open(PREDICTION_FILE_PATH) as annotationFile:
        lines = annotationFile.readlines()
        for line in lines:
            main_colony = line.split() #elements[] = [class, x, y, width, height, confidence]
            #determine height 
            ratio = abs((float(main_colony[4]) / float(main_colony[3])) - 1 )
            #checks: colony is within petri dish     AND     colony is large enough     AND     colony is small enough    AND    ratio is small enough (colony is roound enough)
            # if (distance(float(main_colony[1]),float(elements[2])) < .4) and float(elements[4]) > MIN_SIZE and float(elements[4]) < MAX_SIZE and ratio < MAXIMUM_RATIO:
            if(distance(float(main_colony[1]),float(main_colony[2])) < PETRI_DISH_RADIUS and float(main_colony[5]) > MIN_SELECTION_CONFIDENCE):
                bad = False
                for line in lines: 
                    more_elements = line.split()
                    #assess the closest colony to the current colony, if it is closer than MIN_DISTANCE, don't display the current colony
                    if (distance(float(more_elements[1]),float(more_elements[2]), float(main_colony[1]), float(main_colony[2])) < MIN_DISTANCE and #distance to colony
                        distance(float(more_elements[1]),float(more_elements[2]), float(main_colony[1]), float(main_colony[2])) != 0.0 and         #make sure it's not the same colony
                        float(main_colony[5]) > MIN_DISCRIMINATION_CONFIDENCE):                                                                 #make sure the colony prediction is confident enough to be used for discrimination
                        bad = True
                    
                    # if bad:
                        


if __name__ == "__main__":
    
    DISPLAY_IMAGE_FOLDER_PATH = 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/images/realTest_v3/unprocessed/WIN_20231113_12_54_17_Pro.jpg'
    PREDICTION_FOLDER_PATH =    'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/runs/detect/predict6/labels/WIN_20231113_12_54_17_Pro.txt'
    DISPLAY_TIME = 1000
    discriminate(PREDICTION_FOLDER_PATH, DISPLAY_TIME)






    
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



                        # next_closest_colony_x = int(float(more_elements[1]) * img.shape[1])
                        # next_closest_colony_y = int(float(more_elements[2]) * img.shape[0])
                        # next_closest_colony_r = int(float(more_elements[3]) * img.shape[1] / 2) 

