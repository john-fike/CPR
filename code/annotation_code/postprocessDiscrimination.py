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
def distance(x0, y0, r1=0, x1=.5, y1=.5, r2=0):
    x_dist = abs(x0 - x1) ** 2
    y_dist = abs(y0 - y1) ** 2
    distance = (x_dist + y_dist) ** .5
    distance = distance - (r1 + r2)
    return (x_dist + y_dist) ** .5


# -----------------------------------------------DISCRIMINATION VARIABLES----------------------
IMAGE_PATH = './images/realTest_v3/unprocessed/WIN_20231113_12_54_17_Pro.jpg'
ANNOTATION_PATH = 'C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/runs/detect/predict6/labels/WIN_20231113_12_54_17_Pro.txt'

MIN_DISTANCE = .00
MIN_CONFIDENCE = .0
MARGIN = 1

# -----------------------------------------------STATS----------------------

#images containing doublets 
doublets = [40, 50, 65, 80, 90, 100, 105, 110, 115, 120, 6, 12, 18, 21, 24, 39, 51, 54, 63, 66, 72, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120]

# (x,y) pixel positions
x_avg = []
y_avg = []
doublet_x_avg = []
doublet_y_avg = []

# sort x_avg and y_avg in ascending order
x_avg.sort()
y_avg.sort()

erosion_iterations = 20
image_intensities_vs_erosion_iterations = []
doublet_image_intensities_vs_erosion_iterations = []

# Load your image
# img = cv2.imread('./realTest_v1/processed/WIN_20231024_10_15_42_Pro.jpg')
img = cv2.imread(IMAGE_PATH)

# Check if the image was loaded successfully
if img is None:
    print("Error: Could not read image file")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
gray_blur = cv2.GaussianBlur(gray, (3, 3), 2)

img_width = img.shape[1]
img_height = img.shape[0]

def discriminate(IMAGE_PATH, ANNOTATION_PATH):
    with open (ANNOTATION_PATH, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 3):
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
                    # if the colony meets all the criteria, display it  
                    x, y, width, height = int(float(elements[1]) * img_width), int(float(elements[2]) * img_height), int(float(elements[3]) * img_width * MARGIN), int(float(elements[4]) * img_height * MARGIN)
                    cropped_image = img[int(y-height) : int(y+height) , int(x-width) : int(x+width)]
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
                        binary_image_sums_temp = []
                        for l in range(erosion_iterations):
                            eroded_binary_image = cv2.erode(binary_image, np.ones((2,2), np.uint8), iterations=l)
                            binary_image_sum = np.sum(eroded_binary_image)          #sum of all pixels in the image
                            print("Erosion iterations:" , l)
                            print("Binary sum: " , binary_image_sum)
                            binary_image_sum = binary_image_sum / (width * height)  #normalize
                            print("Normalized binary sum: " , binary_image_sum)
                            binary_image_sums_temp.append(binary_image_sum)
                        
                        # eroded_binary_image = cv2.erode(binary_image, np.ones((2,2), np.uint8), iterations=10)
                        # very_eroded_binary_image = cv2.erode(binary_image, np.ones((2,2), np.uint8), iterations=20)

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
                        average_row_position = np.dot(row_positions, row_sums) / (total_row_sum * width * 2)
                        average_column_position = np.dot(column_positions, column_sums) / (total_column_sum * height * 2)


                        # -----------------------------------------------APPEND TO ARRAYS----------------------
                        # check if i is within an array of ints
                        if(i in doublets):
                            doublet_image_intensities_vs_erosion_iterations.append(binary_image_sums_temp)

                            doublet_x_avg.append(average_column_position)
                            doublet_y_avg.append(average_row_position)
                        else:
                            # doublet_image_intensities_vs_erosion_iterations.append(binary_image_sums_temp)
                            image_intensities_vs_erosion_iterations.append(binary_image_sums_temp)

                            x_avg.append(average_column_position)
                            y_avg.append(average_row_position)


                        # -----------------------------------------------DISPLAY----------------------
                        display_time = 500

                        resized_image = cv2.resize(cropped_image, (512, 512))
                        title = 'ORIGINAL ' + str(i)
                        cv2.imshow(title, resized_image)
                        cv2.waitKey(display_time)

                        resized_image = cv2.resize(binary_image, (512, 512))
                        title = 'BINARY ' 
                        print(title)
                        cv2.imshow(title, resized_image)
                        cv2.waitKey(display_time)

                        # resized_image = cv2.resize(eroded_binary_image, (512, 512))
                        # title = 'ERODED BINARY '
                        # print(title)
                        # cv2.imshow(title, resized_image)
                        # cv2.waitKey(display_time)

                        # resized_image = cv2.resize(very_eroded_binary_image, (512, 512))
                        # title = 'VERY ERODED BINARY ' 
                        # print(title)
                        # cv2.imshow(title, resized_image)
                        # cv2.waitKey(display_time)

                        # resized_image = cv2.resize(very_very_eroded_binary_image, (512, 512))
                        # title = 'VERY VERY ERODED BINARY ' + str(i) + " " + str(very_very_eroded_binary_image_sum/1000)
                        # print(title)
                        # cv2.imshow(title, resized_image)
                        # cv2.waitKey(display_time)


                        cv2.waitKey(0)   
                        cv2.destroyAllWindows()




    # -----------------------------------------------PLOT---------------------- 

    # binary_image_sums_temp = 
    #    intensity per         
    #   erosion iteration       
    #           |             
    #          [*, *, *, ... 20]
    # image -- [*, *, *, ... 20]
    #          [*, *, *, ... 20]
    #                         |   
    #                         |
    #                20 erosion iterations


    # #plot a graph of image sums vs. erosion iterations
    x = np.arange(0, erosion_iterations, 1)
    
    # #Transpose & calculate the average of each column
    transposed_binary_image_sums = list(map(list, zip(*image_intensities_vs_erosion_iterations)))
    doublet_transposed_binary_image_sums = list(map(list, zip(*doublet_image_intensities_vs_erosion_iterations)))

    # find average intensity per erosion iteration, plot
    column_averages = [sum(column) / len(column)*1000 for column in transposed_binary_image_sums]
    print(column_averages)
    y = column_averages
    plt.plot(x, y, color='g')

    column_averages = [sum(column) / len(column)*1000 for column in doublet_transposed_binary_image_sums]
    y = column_averages
    print(column_averages)
    plt.plot(x, y, color='g')

    plt.title('Image Intensity vs. Erosion Iterations')
    plt.xlabel('Erosion Iterations')
    plt.ylabel('Image Intensity')
    plt.show()
    plt.waitforbuttonpress()
    plt.close()

    # x_avg = sum(x_avg) / len(x_avg)
    # y_avg = sum(y_avg) / len(y_avg)

    # doublet_x_avg = sum(doublet_x_avg) / len(doublet_x_avg)
    # doublet_y_avg = sum(doublet_y_avg) / len(doublet_y_avg)

    # # print average value in x_avg
    # print("x_avg" , sum(x_avg) / len(x_avg))
    # print("y_avg" , sum(y_avg) / len(y_avg))

    # print("doublet_x_avg" , sum(doublet_x_avg) / len(doublet_x_avg))
    # print("doublet_y_avg" , sum(doublet_y_avg) / len(doublet_y_avg))
    
    
    # plt.plot(x_avg, y_avg, 'ro')
    # plt.plot(doublet_x_avg, doublet_y_avg, 'bo')
    # plt.show()
# maybe like find most confident 5 colonies, and set that as the upper threshold + like 1.2 std dev
discriminate(IMAGE_PATH, ANNOTATION_PATH)