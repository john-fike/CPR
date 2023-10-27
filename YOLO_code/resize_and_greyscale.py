import os
import cv2

filepath = './realTest_v1/'
unprocessedFiles = os.path.join(filepath, 'unprocessed')
processedFiles = os.path.join(filepath, 'processed')


for file in os.listdir(unprocessedFiles):
    # read in image and convert to grayscale
    img = cv2.imread(os.path.join(unprocessedFiles, file) , cv2.IMREAD_GRAYSCALE)
    # resize image
    img = cv2.resize(img, (640, 640))
    # save image as a jpg
    cv2.imwrite(os.path.join(processedFiles, file.split('.')[0] + '.jpg') , img)
