import cv2
import numpy as np
import os

def add_hough_circles(image_folder_path, prediction_output_path, margin = 1, output_confidence = ".15", display=False, display_time=5000):
    print("Adding hough circles for folder: " + image_folder_path)
    for image in os.listdir(image_folder_path):
        print("Reading image file: " + os.path.join(image_folder_path, image))
        img = cv2.imread(os.path.join(image_folder_path, image) , cv2.IMREAD_COLOR)
        if img is None:
            print("Error: Could not load the image")
            exit()

        # make image grayscale if needed 
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        print(os.path.split((os.path.join(image_folder_path, image)))[1])
        output_file = os.path.join(prediction_output_path, image.split('.')[0] + ".txt") #fix 
        print("Output file: ", output_file)
        # output_file = os.path.join(prediction_output_path, img.split('/')[-1].split('.')[0] + '.txt')

        # detect circles
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 60)                                                                                        ##PARAM
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=15, minRadius=5, maxRadius=30)   ##PARAM
        
        #clear contents of output.txt
        open(output_file, 'w').close()

        #plot circles
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

        if display:
            img = cv2.imread(os.path.join(image_folder_path, image) , cv2.IMREAD_COLOR)

        for (x, y, r) in circles:
            image_width = img.shape[1]
            image_height = img.shape[0]    
            with open (output_file, 'a') as f:
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

#iterate through a foler
if __name__ == "__main__":
    image_folder_path = './images/realTest_v3/unprocessed'
    prediction_output_path = './output/hough'
    add_hough_circles(image_folder_path = image_folder_path, prediction_output_path = prediction_output_path, output_confidence=".15", display=True, display_time=5000)



