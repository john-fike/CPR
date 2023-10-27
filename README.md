
# 03CPR Bacteria Picking Robot Software

Developed by: William Culhane, Sarah Dolan, and John Fike

## Table of Contents
- [Description](#description)
- [YOLO Code](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Description

Software developed for the bacteria Colony Picking Robot developed by ECE/ME capstone team 03CPR. This software is designed to automatically detect bacteria in petri dishes, and then sample them up using a robotic gantry system.


## [YOLO Code](code/YOLO_code) 
Code for creating, testing, and [YOLOv8](https://docs.ultralytics.com/models/yolov8/#key-features) models
* train.py: trains the model on the training set using data.yaml. Training datasets typically creating using roboflow.com
* valid.py: tests the model on the validation set using data.yaml
* predict.py: runs the model on images in a folder. Saves annotated images without label/confidence and saves a .txt file with label/confidence/bounding box coordinates
* resize_and_greyscale.py: resizes to 640x640 and converts to greyscale
>    -folder 
      >          \ -unprocessed (original images)
>            |-processed (resized and greyscale images)
## Annotation Code 
### Code for visualizing and refining YOLO annotations


## Test Code
* cannyEdge.py - creates and displays canny edge of an image. 
* testErosion.py - creates and displays erosion of an image. 
* testMinima.py - creates and displays minima of an image. Just a slightly different application of erosion. Does not work very well. 
* testSegmentBacteria.py - uses YOLO detection annotations to create segmentation annotations. Does not work very well either. 




## Models 

agarV1.pt
trained on AGAR dataset, 640px, 100epoch

agar_v2.pt
trained on AGAR dataset, 70% greyscale, 190epoch (1000 w/ 50 patence)

norbert_v2.pt
trained on harbinger, manually created norbert partitions

agar_v3.pt
trained on agar and norb, 100% greyscale, fixed dumb label thing
1000 epoch w/ 50 patience 

norbert_v3.pt
CURRENTLY IN USE
100% greyscale duplicate partition. 60% hue shift, 1000 epoch w/ 50 patience
bounding box blur and noise, 2.5px