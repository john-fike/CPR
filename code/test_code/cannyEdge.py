import cv2 
import numpy as np 
  
# Reading the input image 
img = cv2.imread('C:/Users/John Fike/OneDrive/Documents/Visual Studio 2022/CPR/runs/detect/predict86/8-31-16_10^3.jpg' , cv2.IMREAD_GRAYSCALE)
  
# Taking a matrix of size 5 as the kernel 
kernel = np.ones((5, 5), np.uint8) 
  
# The first parameter is the original image, 
# kernel is the matrix with which image is 
# convolved and third parameter is the number 
# of iterations, which will determine how much 
# you want to erode/dilate a given image. 
img_erosion = cv2.erode(img, kernel, iterations=1) 
img_dilation = cv2.dilate(img, kernel, iterations=1) 
img_canny = cv2.Canny(img_dilation, 100, 300)

cv2.imshow('Input', img) 
cv2.imshow('Erosion', img_erosion) 
cv2.imshow('Dilation', img_dilation) 
cv2.imshow('Canny', img_canny)
  
cv2.waitKey(0) 