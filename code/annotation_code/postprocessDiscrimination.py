import cv2
import numpy as np
import skimage.morphology as morph
from scipy.stats import norm
import matplotlib.pyplot as plt

# -----------------------------------------------MINIMA FUNCTIONS----------------------
def imextendedmin(img, h):
    marker = img + h
    h_min = cv2.erode(marker, h)
    return (img == h_min)

def imimposemin(img, minima):
    marker = np.full_like(img, np.inf)
    marker[minima] = 0
    mask = np.minimum((img + 1), marker)
    return cv2.erode(marker, mask)
#--------------------------------------------------------------------------------------  DISTANCE FUNCTION

#function to determine the distance from the center of the image to the center of the box
def distance(x0, y0, x1=.5, y1=.5):
    x_dist = abs(x0 - x1) ** 2
    y_dist = abs(y0 - y1) ** 2
    return (x_dist + y_dist) ** .5


# Load your image
# img = cv2.imread('./realTest_v1/processed/WIN_20231024_10_15_42_Pro.jpg')
img = cv2.imread('./realTest_v1/unprocessed/WIN_20231024_10_19_20_Pro.jpg')

# Check if the image was loaded successfully
if img is None:
    print("Error: Could not read image file")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
# gray_blur = cv2.GaussianBlur(gray, (3, 3), 2)




# -----------------------------------------------DISCRIMINATION VARIABLES----------------------
img_width = img.shape[1]
img_height = img.shape[0]
MIN_DISTANCE = .00
MIN_CONFIDENCE = .0
MARGIN = 1

# -----------------------------------------------ANALSIS----------------------
yolo_found = 0
found_circles = 0

binary_image_sum_avg = []
eroded_binary_image_sum_avg = []
very_eroded_binary_image_sum_avg = []
very_very_eroded_binary_image_sum_avg = []

doublet_binary_image_sum_avg = []
doublet_eroded_binary_image_sum_avg = []
doublet_very_eroded_binary_image_sum_avg = []
doublet_very_very_eroded_binary_image_sum_avg = []

doublets = [74,80,83,84,85,86,90,91,92,93,94,95,96,97]



with open ('C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/runs/detect/predict88/labels/WIN_20231024_10_19_20_Pro.txt', 'r') as file:
    lines = file.readlines()
    for i in range(71, len(lines)):
        line = lines[i]
        elements = line.split() #elements[] = [class, x, y, width, height, confidence]

        # -----------------------------------------------DISCRIMINATION----------------------
        if(distance(float(elements[1]),float(elements[2])) < .35):
            bad = False
            for line in lines: 
                more_elements = line.split()
                # assess the closest colony to the current colony, if: it is closer than MIN_DISTANCE, isn't the same colony, and has a confidence greater than MIN_CONFIDENCE, don't display the current colony
                if (distance(float(more_elements[1]),float(more_elements[2]), float(elements[1]), float(elements[2])) < MIN_DISTANCE and distance(float(more_elements[1]),float(more_elements[2]), float(elements[1]), float(elements[2])) != 0.0) and float(more_elements[5]) > MIN_CONFIDENCE:
                    bad = True
                    
            # -----------------------------------------------CROPPING----------------------
            if not bad:
            #     # if the colony meets all the criteria, display it  
                x, y, width, height = int(float(elements[1]) * img_width), int(float(elements[2]) * img_height), int(float(elements[3]) * img_width * MARGIN), int(float(elements[4]) * img_height * MARGIN)
                cropped_image = img[int(y-height) : int(y+height) , int(x-width) : int(x+width)]
                yolo_found = yolo_found + 1
                if(cropped_image is None):
                    print("Error: Could not read image file")
                
                # -----------------------------------------------CIRCLE DETECT----------------------
                else:
                    gray_cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    canny_cropped_image = cv2.Canny(cropped_image, 100, 300)
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
                    eroded_binary_image = cv2.erode(binary_image, np.ones((2,2), np.uint8), iterations=3)
                    very_eroded_binary_image = cv2.erode(binary_image, np.ones((2,2), np.uint8), iterations=8)
                    very_very_eroded_binary_image = cv2.erode(binary_image, np.ones((2,2), np.uint8), iterations=12)



                    # -----------------------------------------------DRAW CIRCLES----------------------
                    circles = cv2.HoughCircles(gray_cropped_image, cv2.HOUGH_GRADIENT, dp=1, minDist= 10, param1=4, param2=10, minRadius=1, maxRadius=0)                    
                    # if circles is not None:
                    #     circles = np.uint16(np.around(circles))
                    #     for i in circles[0, :]:
                    #         cv2.circle(cropped_image, (i[0], i[1]), i[2], (0, 255, 0), 1)
                    #         #print the center of the circle
                    #         print("x: " + str(i[0]) + " y: " + str(i[1]))
                    #     found_circles = found_circles + 1
                            

                            #display only the first circle
                            # i = circles[0, 0]
                            # cv2.circle(cropped_image, (i[0], i[1]), i[2], (0, 255, 0), 1)
                            # Resize the image to 512x512


                    # -----------------------------------------------IMAGE INTENSITY SUMS----------------------
                    binary_image_sum = np.sum(binary_image)
                    eroded_binary_image_sum = np.sum(eroded_binary_image)
                    very_eroded_binary_image_sum = np.sum(very_eroded_binary_image)
                    very_very_eroded_binary_image_sum = np.sum(very_very_eroded_binary_image)
                    #-----------------------------------------------AVERAGE (x,y) PIXEL POSITIONS----------------------
                    row_sums = np.sum(binary_image, axis=1)
                    column_sums = np.sum(binary_image, axis=0)

                    # Calculate row and column positions
                    row_positions = np.arange(binary_image.shape[0])
                    column_positions = np.arange(binary_image.shape[1])

                    # Compute the total row and column sums
                    total_row_sum = np.sum(row_sums)
                    total_column_sum = np.sum(column_sums)

                    # Calculate the average row and column positions
                    average_row_position = np.dot(row_positions, row_sums) / total_row_sum
                    average_column_position = np.dot(column_positions, column_sums) / total_column_sum

                    print(f"Average Row Position: {average_row_position}")
                    print(f"Average Column Position: {average_column_position}")

                    # check if i is within an array of ints
                    if(i in doublets):
                        doublet_binary_image_sum_avg.append(binary_image_sum)
                        doublet_eroded_binary_image_sum_avg.append(eroded_binary_image_sum)
                        doublet_very_eroded_binary_image_sum_avg.append(very_eroded_binary_image_sum)   
                        doublet_very_very_eroded_binary_image_sum_avg.append(very_very_eroded_binary_image_sum)
                    else:
                        binary_image_sum_avg.append(binary_image_sum)
                        eroded_binary_image_sum_avg.append(eroded_binary_image_sum)
                        very_eroded_binary_image_sum_avg.append(very_eroded_binary_image_sum)
                        very_very_eroded_binary_image_sum_avg.append(very_very_eroded_binary_image_sum)


                    # -----------------------------------------------DISPLAY----------------------
                    display_time = 1500

                    resized_image = cv2.resize(cropped_image, (512, 512))
                    title = 'ORIGINAL ' + str(i)
                    cv2.imshow(title, resized_image)
                    cv2.waitKey(display_time)

                    resized_image = cv2.resize(binary_image, (512, 512))
                    title = 'BINARY ' + str(i) +  " " + str(binary_image_sum/1000)
                    print(title)
                    cv2.imshow(title, resized_image)
                    cv2.waitKey(display_time)

                    resized_image = cv2.resize(eroded_binary_image, (512, 512))
                    title = 'ERODED BINARY ' + str(i) + " " + str(eroded_binary_image_sum/1000)
                    print(title)
                    cv2.imshow(title, resized_image)
                    cv2.waitKey(display_time)

                    resized_image = cv2.resize(very_eroded_binary_image, (512, 512))
                    title = 'VERY ERODED BINARY ' + str(i) + " " + str(very_eroded_binary_image_sum/1000)
                    print(title)
                    cv2.imshow(title, resized_image)
                    cv2.waitKey(display_time)

                    resized_image = cv2.resize(very_very_eroded_binary_image, (512, 512))
                    title = 'VERY VERY ERODED BINARY ' + str(i) + " " + str(very_very_eroded_binary_image_sum/1000)
                    print(title)
                    cv2.imshow(title, resized_image)
                    cv2.waitKey(display_time)



                    cv2.destroyAllWindows()




    # -----------------------------------------------PLOT---------------------- 
    #plot a graph of image sums vs. erosion iterations
    x = [0, 1, 2, 3]
    y = [np.mean(binary_image_sum_avg), np.mean(eroded_binary_image_sum_avg), np.mean(very_eroded_binary_image_sum_avg), np.mean(very_very_eroded_binary_image_sum_avg)]
    

    z = [np.mean(doublet_binary_image_sum_avg), np.mean(doublet_eroded_binary_image_sum_avg), np.mean(doublet_very_eroded_binary_image_sum_avg), np.mean(doublet_very_very_eroded_binary_image_sum_avg)]


    #bar graph plot boublet avg next to normal avg against erosion iterations
    plt.plot(x, y, color='g')
    plt.plot(x, z, color='r')
    plt.title('Average Image Intensity vs. Erosion Iterations')
    plt.xlabel('Erosion Iterations')
    plt.ylabel('Average Image Intensity')
    plt.show()


print("yolo found" , yolo_found)
print("doublet discrimination found only " , found_circles)
