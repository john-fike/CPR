
# 03CPR Bacteria Picking Robot Software
*Developed by: [William Culhane](https://wculhane.com/), Sarah Dolan, and [John Fike](https://github.com/john-fike)*

## Table of Contents
- [Description](#description)
- [YOLO](YOLO)
- [Annotation Code](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Description

Software developed for the bacteria Colony Picking Robot developed by ECE/ME capstone team 03CPR. This software is designed to automatically detect bacteria in petri dishes, and then sample them up using a robotic gantry system.


## YOLO
Code for creating, testing, and [YOLOv8](https://docs.ultralytics.com/models/yolov8/#key-features) models
* [train.py](YOLO_code/train.py): trains the model on the training set using data.yaml. Training datasets typically creating using roboflow.com
* [valid.py](YOLO_code/valid.py): tests the model on the validation set using data.yaml
* [predict.py](YOLO_code/predict.py): runs the model on images in a folder. Saves annotated images without label/confidence and saves a .txt file with label/confidence/bounding box coordinates
* [resize_and_greyscale.py](YOLO_code/resize_and_greyscale.py): resizes to 640x640 and converts to greyscale

   |--folder 
        |--unprocessed (original images)
        |--processed (resized and greyscale images)


## Annotation Code 
Code for visualizing and refining YOLO annotations
* [postprocessDiscrimination.py](annotation_code/postprocessDiscrimination.py): takes in a .txt file with label/confidence/bounding box coordinates and provides various options for determining if an annotation contains a doublet colony. 
* [showpredictions.py](annotation_code/showpredictions.py): takes in .txt files in a folder, and displays the annotated images with label/confidence/bounding box coordinates. Currently provides various variables for filtering out annotations with potentially undesirable colonies. 

## Test Code
* [cannyEdge.py](test_code/cannyEdge.py) - creates and displays canny edge of an image. 
* [testErosion.py](test_code/testErosion.py) - creates and displays erosion of an image. 
* [testMinima.py](test_code/testMinima.py) - creates and displays minima of an image. Just a slightly different application of erosion. Does not work very well. 
* [testSegmentBacteria.py](test_code/testSegmentBacteria.py) - uses YOLO detection annotations to create segmentation annotations. Does not work very well either. 


## Models 
** only one model added because they are big ** 
norbert_v3.pt
CURRENTLY IN USE
100% greyscale duplicate partition. 60% hue shift, 1000 epoch w/ 50 patience
bounding box blur and noise, 2.5px

[License: GNU GPLv3](COPYING.md)
